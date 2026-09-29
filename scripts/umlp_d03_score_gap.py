"""UMLP-D0.3: how far below the top 5 the frozen twin leaves its missed 2wiki secondary golds, in the units of v2.2A's
bound, and what v2.2A actually moved (configs/umlp_d03_score_gap_feasibility.yaml).

    python scripts/umlp_d03_score_gap.py --stage score   # V2_GATE compile + forward pass of the frozen twin s0-2 and v2.2A s0, integrity-checked
    python scripts/umlp_d03_score_gap.py --stage read    # gaps, feasibility, learned offsets, readings -> record.json (+ tables.md)
    python scripts/umlp_d03_score_gap.py --stage file --date 2026_09_29

Analysis only: no parameter is updated (eval mode, no_grad), held rows are removed before compilation, v2.2A is not altered.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V22A = _load("umlp_v22a", "umlp_v22a_offset.py")
V21 = V22A.V21
D01 = _load("umlp_d01", "umlp_d01_feature_readout.py")

CONFIG = ROOT / "configs" / "umlp_d03_score_gap_feasibility.yaml"
FROZEN_CONFIGS = {   # byte-pinned for the whole phase (sha256 when D0.3 was declared)
    "universal_v2.yaml": "bce0d1e92de4f184f09f0770fd378e2fa63bf469fdf31bb904d7471a11eeb562",
    "umlp_d0_2wiki_diagnostics.yaml": "c48720d84c6f4866ebd923bd14fb0203230785d814217e6ed2e3560e8e77f28d",
    "umlp_d01_2wiki_feature_readout.yaml": "9a9b878925c401dc68d158bdfa65a8780d7154d9955fb747b91509c779d61774",
    "umlp_d02_joint_separability.yaml": "87ab5e03546dd5cca9277dad009dfa87a34a6f35bcc73f76bb663ff3dcc5b76c",
    "umlp_v21_coverage_objective.yaml": "cd786f7454c8734de4f8192132021a49a6b60eedad08aaa4c05051164ed0ccf3",
    "umlp_v22a_linear_offset.yaml": "4cc2952d055e0386a3a6f6579d3dd628de01b0f8308fea8ebe07332d842e2691",
}
OUT = ROOT / "outputs" / "umlp_d03"
SIDECAR = OUT / "gaps_2wiki_gate.npz"
RECORD = OUT / "record.json"
TABLES = OUT / "tables.md"
DOC = ROOT / "docs" / "UMLP_D03_SCORE_GAP_FEASIBILITY.md"
NAME = "2wiki"
TWINS = [V21.key_of(V22A.BASE, s) for s in (0, 1, 2)]
TWIN = TWINS[0]
OFF = V21.key_of(V22A.NEW, 0)
TAU, EPS = V22A.TAU, 1e-6
KEEP_TOP = 20
THREADS = 6            # the stored eval records' and D0.1's thread count
SAT = 0.9
Z_TOL = 1e-5
KEY = 1_000_000        # (query, local) -> query * KEY + local; pools are far smaller
RESAMPLES, SEED, LEVEL = 1000, 0, 95
B_SHARE, A_SHARE, A_BAD, C_SHARE = 0.25, 2 / 3, 0.5, 2 / 3
CATS = ("correct", "wrong", "none", "reversed", "far_too_small", "bad")
STRATA = ("all", "gold_count=2", "gold_count>=3", "h1", "h2", "h3", "none", "G_miss_mp", "G_miss_both")
METRICS = V21.METRICS
LF = chr(10)
sha256_file, utc = V21.sha256_file, V21.utc


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def frozen_unchanged() -> None:
    for n, h in FROZEN_CONFIGS.items():
        if sha256_file(ROOT / "configs" / n) != h:
            raise SystemExit(f"configs/{n} is not its pinned bytes; hard stop")


def verify_pins(decl: dict) -> None:
    inp = decl["inputs"]
    for item in inp.values():
        if isinstance(item, dict) and "sha256" in item and sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise SystemExit(f"{item['path']}: not the pinned sha256; hard stop")
    for key, digest in inp["twin_checkpoints"].items():
        if sha256_file(V22A.V2_FITS / f"{key}.pt") != digest:
            raise SystemExit(f"{key}.pt: not the pinned sha256; hard stop")
    frozen_unchanged()


# ── the declared quantities (pure; the tests exercise these) ─────────────────


def query_units(shat: np.ndarray, rank: np.ndarray, gold: np.ndarray, eps: float = EPS) -> dict | None:
    """One query's kept rows under one twin seed (every in-pool gold and the twin's top 20). None unless the twin's
    rank 1 is a gold and another in-pool gold ranks below 5. N_disp is ordered hardest (best rank) to nearest."""
    if not (gold & (rank == 1)).any():
        return None
    miss = np.flatnonzero(gold & (rank > 5))
    if miss.size == 0:
        return None
    r5 = np.flatnonzero(rank == 5)
    if r5.size != 1:
        raise ValueError("the twin's rank-5 candidate is not among the kept rows")
    kth = float(shat[r5[0]])
    disp = np.flatnonzero(~gold & (rank >= 2) & (rank <= 5))
    disp = disp[np.argsort(rank[disp], kind="stable")]
    return {"miss": miss, "kth": kth, "rank5_is_gold": bool(gold[r5[0]]), "required": np.maximum(0.0, kth - shat[miss] + eps),
            "disp": disp, "hardest": int(disp[0]) if disp.size else -1, "nearest": int(disp[-1]) if disp.size else -1}


def pair_required(shat: np.ndarray, g: int, n: int, eps: float = EPS) -> float:
    return max(0.0, float(shat[n] - shat[g]) + eps)


def movement(delta: np.ndarray, rank_off: np.ndarray, g: int, n: int, r: float) -> dict:
    """v2.2A's learned movement of one (gold, displacer) pair against the correction r it required."""
    m = float(delta[g] - delta[n])
    rev = bool(rank_off[g] < rank_off[n])
    small = (0 < m < r / 2) and not rev
    return {"m": m, "correct": m > 0, "wrong": m < 0, "none": m == 0, "reversed": rev, "far_too_small": small, "bad": (m <= 0) or small}


def query_fc_requirement(shat: np.ndarray, gold: np.ndarray, gold_total: int, eps: float = EPS) -> tuple[float, bool]:
    """(required_fc, reachable by any reranker). The golds must all clear the (6 - G_in)-th highest non-gold."""
    g_in = int(gold.sum())
    if gold_total != g_in or g_in > 5:
        return float("nan"), False
    ng = np.sort(shat[~gold])[::-1]
    k = 6 - g_in
    if ng.size < k:
        return 0.0, True
    return max(0.0, float(ng[k - 1] - shat[gold].min()) + eps), True


def groups(z: dict, twin: str) -> dict[str, np.ndarray]:
    """D0.1's G_miss / G_miss_mp / G_miss_both / N_disp on this sidecar's rows (checked against D0.1's groups_for_seed)."""
    rank, gold, qi = z[f"{twin}/rank"], z["is_gold"], z["query"]
    top_gold = np.zeros(z["row"].size, dtype=bool)
    top_gold[qi[gold & (rank == 1)]] = True
    analysed = top_gold[qi]
    g = {"G_miss": analysed & gold & (rank > 5), "N_disp": analysed & ~gold & (rank >= 2) & (rank <= 5)}
    g["G_miss_mp"] = g["G_miss"] & (z["gnn_top5_votes"] >= 2)
    g["G_miss_both"] = g["G_miss"] & (z["gnn_top5_votes"] <= 1)
    return g


def check_groups_against_d01(z: dict, z1: dict, gnn_keys: list[str]) -> dict:
    """Every declared group, as a set of (query, local), equal to D0.1's groups_for_seed on its own sidecar."""
    def keys(d, m):
        return set((d["query"][m].astype(np.int64) * KEY + d["local"][m]).tolist())
    diff = {}
    for twin in TWINS:
        g = groups(z, twin)
        g1, _ = D01.groups_for_seed(z1, twin, gnn_keys)
        for name in ("G_miss", "G_miss_mp", "G_miss_both", "N_disp"):
            a, b = keys(z, g[name]), keys(z1, g1[name])
            if a != b:
                diff[f"{twin}/{name}"] = {"d03_only": len(a - b), "d01_only": len(b - a)}
    return diff


# ── stage: score ─────────────────────────────────────────────────────────────


def d01_gold_table(decl: dict, gi: np.ndarray) -> dict:
    gnn = D01.load_declaration()["arms_scored"]["mp_reference"]
    with np.load(ROOT / decl["inputs"]["d01_sidecar"]["path"]) as z:
        if not np.array_equal(z["row"], gi):
            raise SystemExit("the D0.1 sidecar rows are not these V2_GATE rows; refusing")
        g = z["is_gold"]
        key = z["query"][g].astype(np.int64) * KEY + z["local"][g]
        order = np.argsort(key, kind="stable")
        return {"key": key[order], "votes": np.sum([z[f"{k}/rank"][g][order] <= 5 for k in gnn], axis=0),
                **{f"{k}/rank": z[f"{k}/rank"][g][order] for k in TWINS},
                "gold_total": z["gold_total"], "pool_size": z["pool_size"], "first_support_STRUCT": z["first_support_STRUCT"]}


def stage_score(decl: dict, log=print) -> None:
    if SIDECAR.exists():
        log(f"{SIDECAR} exists; not rescored")
        return
    torch, U = V21.load_runner(THREADS)
    from mp_retrieval.m3b_models import segment_zscore
    verify_pins(decl)
    cfg, cfg_m3b, cfg_h = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    m3b_contract = U.M3B_RUN.load_script("m3b_contract")
    contexts, handles, pkg, bank = U.open_contexts_v2(cfg_m3b, [NAME], m3b_compile)
    context, ds = contexts[NAME], handles[NAME]
    models = {}
    for s, k in enumerate(TWINS):
        models[k], _ = V22A.load_base(U, torch, inputs, bank, s)
    rec = json.loads((V22A.FITS / f"{OFF}.json").read_text(encoding="utf-8"))
    if sha256_file(V22A.FITS / f"{OFF}.pt") != rec["state_sha256"]:
        raise SystemExit(f"{OFF}.pt: not the recorded sha256; hard stop")
    base, _ = V22A.load_base(U, torch, inputs, bank, 0)
    off = V22A.make_offset_model(base, inputs["n_scalars"])
    off.load_state_dict(torch.load(V22A.FITS / f"{OFF}.pt", map_location="cpu"))
    if V22A.state_digest(off.base) != rec["base_digest"]:
        raise SystemExit(f"{OFF}: the base inside the offset checkpoint is not the frozen base; hard stop")
    off.eval()
    models[OFF] = off
    half_stored, stored = V21.stored_control(NAME, TWINS)
    with np.load(V22A.EVAL / f"{NAME}.npz") as z:
        off_rows = z["row"]
        stored.update({f"{OFF}/{m}": z[f"{OFF}/{m}"] for m in METRICS})
        stored_mad = z[f"{OFF}/mean_abs_delta"]
    m3a = pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][NAME]["construction"]
    split = cfg_m3b["populations"]["eval_splits"][NAME]
    declared = cfg["m3b_incumbents"]["eval_populations_reused_here"][NAME]
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, NAME, "eval", cfg_m3b, cfg_h, m3a, positions)
    del positions
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]):
        raise SystemExit("not the M3B eval population; refusing")
    half = U.half_labels(NAME, ds, split, pop.ids)
    if not np.array_equal(half, half_stored):
        raise SystemExit("recomputed halves differ from the stored halves; refusing")
    gi = np.flatnonzero(half)                 # held queries leave here, before anything is compiled or scored
    if not np.array_equal(off_rows, gi):
        raise SystemExit("the v2.2A eval rows are not these V2_GATE rows; refusing")
    d01 = d01_gold_table(decl, gi)
    pop.ids, pop.idx, pop.golds = [pop.ids[i] for i in gi], pop.idx[gi], [pop.golds[i] for i in gi]
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    n = len(gi)
    sizes = np.asarray([p.size for p in prep.pools])
    gold_total = np.asarray([g.size for g in pop.golds])
    if not (np.array_equal(sizes, d01["pool_size"]) and np.array_equal(gold_total, d01["gold_total"])):
        raise SystemExit("pool sizes or gold totals differ from the D0.1 sidecar; hard stop")
    chunk = max(1, int(24000 // max(sizes.mean(), 1)))
    columns = inputs["column_indices"]
    cols = defaultdict(list)
    metric = {f"metric/{k}/{m}": np.zeros(n) for k in (TWIN, OFF) for m in METRICS}
    per_q = {k: np.zeros(n) for k in ("sigma", "mu", "delta_abs_max", "delta_abs_mean", "delta_sat_share")}
    gold_in_pool = np.zeros(n, dtype=np.int64)
    zdiff = 0.0
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, chunk):
            idx = np.arange(start, min(start + chunk, n))
            qds, gold_locals = [], []
            for j in idx:
                E = context.nodes.read(prep.pools[j])
                inp = U.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
                compiled = U.compile_query_v2(inp, prep.pools[j], prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
                gl = m3b_compile.gold_local_of(prep.pools[j], pop.golds[j])
                qds.append({"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[j],
                            "seeds": compiled.seeds_local, "gold": gl, "gold_total": int(pop.golds[j].size), "emb": E})
                gold_locals.append(gl)
            batch = U.pack_queries_v2(qds, context)
            ptr = batch.qptr.numpy()
            B = batch.n_queries
            out = {}
            for k, model in models.items():
                s = model(U.arm_view(model, batch, inputs))
                out[k] = s.cpu().numpy().astype(np.float64)
                if k == OFF:
                    s0_hat, delta = model.last
                    out["s0_hat"] = s0_hat.cpu().numpy().astype(np.float64)
                    out["delta32"] = delta.cpu().numpy()
                    out["delta"] = out["delta32"].astype(np.float64)
                else:
                    out[f"{k}/shat"] = segment_zscore(s.unsqueeze(1), batch.node_query, B).squeeze(1).cpu().numpy().astype(np.float64)
                for jj, j in enumerate(idx):
                    r = U.rank_metrics(out[k][ptr[jj]:ptr[jj + 1]], gold_locals[jj], int(pop.golds[j].size))
                    for m in METRICS:
                        if r[m] != stored[f"{k}/{m}"][j]:
                            raise SystemExit(f"query {pop.ids[j]} (row {gi[j]}), {k} {m}: forward pass {r[m]} != stored {stored[f'{k}/{m}'][j]}; hard stop")
                        if k in (TWIN, OFF):
                            metric[f"metric/{k}/{m}"][j] = r[m]
            zdiff = max(zdiff, float(np.abs(out["s0_hat"] - out[f"{TWIN}/shat"]).max()))
            if zdiff > Z_TOL:
                raise SystemExit(f"v2.2A's internal s0_hat differs from the twin s0 z-score by {zdiff}; hard stop")
            for jj, j in enumerate(idx):
                a, b = ptr[jj], ptr[jj + 1]
                ranks = {}
                for k in models:
                    order = np.argsort(-out[k][a:b], kind="stable")
                    rk = np.empty(b - a, dtype=np.int64)
                    rk[order] = np.arange(1, b - a + 1)
                    ranks[k] = rk
                keep = np.zeros(b - a, dtype=bool)
                keep[gold_locals[jj]] = True
                for k in models:
                    keep[ranks[k] <= KEEP_TOP] = True
                loc = np.flatnonzero(keep)
                cols["query"].append(np.full(loc.size, j))
                cols["local"].append(loc)
                cols["is_gold"].append(np.isin(loc, gold_locals[jj]))
                for k in TWINS:
                    cols[f"{k}/score"].append(out[k][a:b][loc])
                    cols[f"{k}/shat"].append(out[f"{k}/shat"][a:b][loc])
                    cols[f"{k}/rank"].append(ranks[k][loc])
                cols[f"{OFF}/s0_hat"].append(out["s0_hat"][a:b][loc])
                cols[f"{OFF}/delta"].append(out["delta"][a:b][loc])
                cols[f"{OFF}/score"].append(out[OFF][a:b][loc])
                cols[f"{OFF}/rank"].append(ranks[OFF][loc])
                gold_in_pool[j] = gold_locals[jj].size
                raw = out[TWIN][a:b]
                per_q["mu"][j], per_q["sigma"][j] = raw.mean(), raw.std()
                ad = np.abs(out["delta32"][a:b])
                per_q["delta_abs_max"][j] = float(ad.max())
                per_q["delta_abs_mean"][j] = float(ad.mean())
                per_q["delta_sat_share"][j] = float((ad >= SAT).mean())
            if (start // chunk) % 5 == 0:
                log(f"   {min(start + chunk, n)}/{n} queries compiled and scored, {time.time() - t0:.0f}s, integrity equal so far (z diff {zdiff:.2e})")
    frozen_unchanged()
    z = {k: np.concatenate(v) for k, v in cols.items()}
    g = z["is_gold"]
    gk = z["query"][g].astype(np.int64) * KEY + z["local"][g]
    if gk.size != d01["key"].size or not np.array_equal(np.sort(gk), d01["key"]):
        raise SystemExit("the in-pool golds are not the D0.1 sidecar's golds; hard stop")
    pos = np.searchsorted(d01["key"], gk)
    rank_mismatch = {k: int((d01[f"{k}/rank"][pos] != z[f"{k}/rank"][g]).sum()) for k in TWINS}
    if any(rank_mismatch.values()):
        raise SystemExit(f"gold ranks differ from the D0.1 sidecar: {rank_mismatch}; hard stop")
    votes = np.zeros(g.size, dtype=np.int64)
    votes[g] = d01["votes"][pos]
    z.update({"gnn_top5_votes": votes, "row": gi.astype(np.int64), "pool_size": sizes, "gold_total": gold_total, "gold_in_pool": gold_in_pool,
              "first_support_STRUCT": d01["first_support_STRUCT"], **{f"q/{k}": v for k, v in per_q.items()}, **metric})
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = SIDECAR.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **z)
    tmp.replace(SIDECAR)
    meta = {"utc": utc(), "queries": n, "candidates_kept": int(z["query"].size), "seconds": round(time.time() - t0, 1), "chunk_queries": chunk,
            "threads": THREADS, "models": list(models), "metric_mismatches": 0, "s0_hat_max_abs_diff": zdiff,
            "gold_rank_mismatch_vs_d01": rank_mismatch, "mean_abs_delta_max_abs_diff_vs_v22a_eval": float(np.abs(per_q["delta_abs_mean"] - stored_mad).max()),
            "integrity": ("every twin seed's and v2.2A's per-query metrics equal the stored arrays on every V2_GATE query; v2.2A's s0_hat equals "
                          "the twin s0 z-score; every gold's twin rank equals the D0.1 sidecar (seeds 0-2)"),
            "sha256": sha256_file(SIDECAR)}
    (OUT / "score_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"score: {n} queries, {meta['candidates_kept']} candidates kept, 0 mismatches, z diff {zdiff:.2e}, {meta['seconds']}s")


# ── stage: read ──────────────────────────────────────────────────────────────


def load_sidecar() -> dict:
    with np.load(SIDECAR) as z:
        return {k: z[k] for k in z.files}


_W: dict[int, np.ndarray] = {}


def _weights(q: int) -> np.ndarray:
    if q not in _W:
        rng = np.random.default_rng(SEED)
        _W[q] = np.stack([np.bincount(rng.integers(q, size=q), minlength=q) for _ in range(RESAMPLES)]).astype(np.float64)
    return _W[q]


def share_ci(flags, q) -> dict:
    """Share of units, and its query-level percentile bootstrap (the share recomputed over the resampled queries' units)."""
    flags, q = np.asarray(flags, dtype=bool), np.asarray(q)
    if flags.size == 0:
        return {"share": None, "low": None, "high": None, "units": 0, "queries": 0}
    uq, inv = np.unique(q, return_inverse=True)
    k = np.bincount(inv, weights=flags.astype(np.float64), minlength=uq.size)
    n = np.bincount(inv, minlength=uq.size).astype(np.float64)
    W = _weights(uq.size)
    lo, hi = np.percentile((W @ k) / (W @ n), [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return {"share": round(float(flags.mean()), 4), "low": round(float(lo), 4), "high": round(float(hi), 4), "units": int(flags.size), "queries": int(uq.size)}


def dist(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [10, 25, 50, 75, 90, 95])
    return {"n": int(v.size), "mean": round(float(v.mean()), 4), **{k: round(float(x), 4) for k, x in zip(("q10", "q25", "median", "q75", "q90", "q95"), q)}}


def collect(z: dict, twin: str, offset: bool) -> tuple[dict, dict, dict]:
    """Units (missed golds), pairs (missed gold x rank 2-5 non-gold) and analysed queries for one twin seed."""
    qi = z["query"]
    order = np.argsort(qi, kind="stable")
    qs, starts = np.unique(qi[order], return_index=True)
    ends = np.append(starts[1:], order.size)
    Us, Ps, Qs = defaultdict(list), defaultdict(list), defaultdict(list)
    shat_all, rank_all, gold_all, votes_all = z[f"{twin}/shat"], z[f"{twin}/rank"], z["is_gold"], z["gnn_top5_votes"]
    for q, a, b in zip(qs, starts, ends):
        idx = order[a:b]
        shat, rank, gold = shat_all[idx], rank_all[idx], gold_all[idx]
        u = query_units(shat, rank, gold)
        if u is None:
            continue
        gt, fs = int(z["gold_total"][q]), int(z["first_support_STRUCT"][q])
        req_fc, reach = query_fc_requirement(shat, gold, gt)
        Qs["q"].append(q)
        Qs["required_fc"].append(req_fc)
        Qs["reachable"].append(reach)
        Qs["gold_total"].append(gt)
        Qs["fs"].append(fs)
        Qs["sigma"].append(z["q/sigma"][q])
        if offset:
            delta, roff = z[f"{OFF}/delta"][idx], z[f"{OFF}/rank"][idx]
            Qs["fc5_off"].append(z[f"metric/{OFF}/full_coverage@5"][q] > 0.5)
        for gpos, req in zip(u["miss"], u["required"]):
            mp = bool(votes_all[idx[gpos]] >= 2)
            for k, v in (("q", q), ("required", req), ("rank5_is_gold", u["rank5_is_gold"]), ("mp", mp), ("gold_total", gt), ("fs", fs),
                         ("has_pair", u["disp"].size > 0), ("twin_rank", rank[gpos])):
                Us[k].append(v)
            for d in ("nearest", "hardest"):
                n = u[d]
                r = pair_required(shat, gpos, n) if n >= 0 else np.nan
                Us[f"r_{d}"].append(r)
                if offset:
                    mv = movement(delta, roff, gpos, n, r) if n >= 0 else None
                    Us[f"m_{d}"].append(mv["m"] if mv else np.nan)
                    Us[f"delta_n_{d}"].append(delta[n] if n >= 0 else np.nan)
                    for c in CATS:
                        Us[f"{c}_{d}"].append(bool(mv[c]) if mv else False)
            if offset:
                Us["delta_g"].append(delta[gpos])
                Us["off_rank"].append(roff[gpos])
                Us["realised"].append(bool(roff[gpos] <= 5))
            for n in u["disp"]:
                r = pair_required(shat, gpos, n)
                for k, v in (("q", q), ("r", r), ("mp", mp), ("gold_total", gt), ("fs", fs)):
                    Ps[k].append(v)
                if offset:
                    mv = movement(delta, roff, gpos, n, r)
                    Ps["m"].append(mv["m"])
                    for c in CATS:
                        Ps[c].append(bool(mv[c]))
    return tuple({k: np.asarray(v) for k, v in d.items()} for d in (Us, Ps, Qs))


def stratum(d: dict, s: str) -> np.ndarray | None:
    gt, fs = d["gold_total"], d["fs"]
    if s == "all":
        return np.ones(gt.size, dtype=bool)
    if s == "gold_count=2":
        return gt == 2
    if s == "gold_count>=3":
        return gt >= 3
    if s in ("h1", "h2", "h3"):
        return fs == int(s[1])
    if s == "none":
        return fs == 0
    if "mp" not in d:
        return None
    return d["mp"].astype(bool) if s == "G_miss_mp" else ~d["mp"].astype(bool)


def summarise_units(u: dict, sel: np.ndarray, offset: bool) -> dict:
    q, req = u["q"][sel], u["required"][sel]
    out = {"units": int(sel.sum()), "queries": int(np.unique(q).size), "required_lift": dist(req),
           "one_sided_lt_tau": share_ci(req < TAU, q), "two_sided_lt_2tau": share_ci(req < 2 * TAU, q), "beyond_2tau": share_ci(req >= 2 * TAU, q),
           "rank5_is_gold": share_ci(u["rank5_is_gold"][sel], q), "no_displacer": int((~u["has_pair"][sel]).sum()),
           "twin_rank": dist(u["twin_rank"][sel])}
    hp = sel & u["has_pair"]
    qh = u["q"][hp]
    for d in ("nearest", "hardest"):
        r = u[f"r_{d}"][hp]
        block = {"required": dist(r), "one_sided_lt_tau": share_ci(r < TAU, qh), "two_sided_lt_2tau": share_ci(r < 2 * TAU, qh)}
        if offset:
            m = u[f"m_{d}"][hp]
            block["learned"] = {"m": dist(m), "m_minus_r": dist(m - r), "m_over_r": dist(m / r), "delta_displacer": dist(u[f"delta_n_{d}"][hp]),
                                "delta_displacer_saturated": share_ci(np.abs(u[f"delta_n_{d}"][hp]) >= SAT, qh),
                                **{c: share_ci(u[f"{c}_{d}"][hp], qh) for c in CATS}}
        out[f"{d}_displacer"] = block
    if offset:
        out["learned_gold"] = {"delta_g": dist(u["delta_g"][sel]), "delta_g_saturated": share_ci(np.abs(u["delta_g"][sel]) >= SAT, q),
                               "realised_entry": share_ci(u["realised"][sel], q), "v22a_rank": dist(u["off_rank"][sel])}
        for name, cut in (("one_sided", u["required"] < TAU), ("beyond_one_sided", u["required"] >= TAU)):
            m = sel & cut
            out["learned_gold"][f"realised_entry_if_{name}"] = share_ci(u["realised"][m], u["q"][m])
    return out


def summarise_pairs(p: dict, sel: np.ndarray, offset: bool) -> dict:
    q, r = p["q"][sel], p["r"][sel]
    out = {"pairs": int(sel.sum()), "queries": int(np.unique(q).size), "required": dist(r),
           "one_sided_lt_tau": share_ci(r < TAU, q), "two_sided_lt_2tau": share_ci(r < 2 * TAU, q)}
    if offset:
        out["learned"] = {"m": dist(p["m"][sel]), **{c: share_ci(p[c][sel], q) for c in CATS}}
    return out


def summarise_queries(Q: dict, sel: np.ndarray, offset: bool) -> dict:
    q, reach = Q["q"][sel], Q["reachable"][sel].astype(bool)
    rq = Q["required_fc"][sel][reach]
    out = {"queries": int(sel.sum()), "reachable_by_reranking": int(reach.sum()), "unreachable_by_any_reranker": int((~reach).sum()),
           "sigma_raw": dist(Q["sigma"][sel]), "required_fc": dist(rq),
           "one_sided_lt_tau": share_ci(rq < TAU, q[reach]), "two_sided_lt_2tau": share_ci(rq < 2 * TAU, q[reach])}
    if offset:
        out["v22a_full_coverage@5"] = share_ci(Q["fc5_off"][sel], q)
    return out


def gold_flows(z: dict, qmask: np.ndarray) -> dict:
    sel = z["is_gold"] & qmask[z["query"]]
    t, o = z[f"{TWIN}/rank"][sel], z[f"{OFF}/rank"][sel]
    fc_t, fc_o = z[f"metric/{TWIN}/full_coverage@5"][qmask], z[f"metric/{OFF}/full_coverage@5"][qmask]
    return {"queries": int(qmask.sum()), "golds": int(sel.sum()), "golds_in_twin_top5": int((t <= 5).sum()), "golds_in_v22a_top5": int((o <= 5).sum()),
            "entered": int(((t > 5) & (o <= 5)).sum()), "left": int(((t <= 5) & (o > 5)).sum()),
            "rank1_gold_left_top5": int(((t == 1) & (o > 5)).sum()), "rank1_gold_moved_off_top1": int(((t == 1) & (o != 1)).sum()),
            "full_coverage@5_gained": int(((fc_t < 0.5) & (fc_o > 0.5)).sum()), "full_coverage@5_lost": int(((fc_t > 0.5) & (fc_o < 0.5)).sum())}


def stage_read(decl: dict, log=print) -> dict:
    verify_pins(decl)
    z = load_sidecar()
    with np.load(D01.SIDECAR) as f:
        z1 = {k: f[k] for k in f.files}
    gnn = D01.load_declaration()["arms_scored"]["mp_reference"]
    diff = check_groups_against_d01(z, z1, gnn)
    if diff:
        raise SystemExit(f"groups differ from D0.1's groups_for_seed: {diff}; hard stop")
    gate = json.loads((ROOT / decl["inputs"]["v22a_gate"]["path"]).read_text(encoding="utf-8"))["paired_v22a_minus_control"]
    per_seed = {}
    for twin in TWINS:
        offset = twin == TWIN
        u, p, Q = collect(z, twin, offset)
        g = groups(z, twin)
        if u["q"].size != int(g["G_miss"].sum()) or int(u["mp"].sum()) != int(g["G_miss_mp"].sum()):
            raise SystemExit(f"{twin}: units are not D0.1's G_miss; hard stop")
        seed = {"analysed_queries": int(Q["q"].size), "missed_golds": int(u["q"].size), "pairs": int(p["q"].size), "strata": {}}
        for s in STRATA:
            su = stratum(u, s)
            block = {"units": summarise_units(u, su, offset)}
            if offset:
                block["pairs_all_displacers"] = summarise_pairs(p, stratum(p, s), offset)
                sq = stratum(Q, s)
                if sq is not None:
                    block["queries_full_coverage"] = summarise_queries(Q, sq, offset)
            seed["strata"][s] = block
        per_seed[twin] = seed
        if offset:
            u0, Q0 = u, Q
            point = {"share_required_ge_2tau": float(np.mean(u["required"] >= 2 * TAU)),
                     "share_required_lt_tau": float(np.mean(u["required"] < TAU)),
                     "share_bad_vs_nearest": float(np.mean(u["bad_nearest"][u["has_pair"]])),
                     "units_with_nearest_displacer": int(u["has_pair"].sum()),
                     "share_mp_required_lt_tau": float(np.mean(u["required"][u["mp"].astype(bool)] < TAU)),
                     "v22a_gate_hit@1_low": gate["hit@1"]["low"], "v22a_gate_full_coverage@5_high": gate["full_coverage@5"]["high"]}
    if point["share_required_ge_2tau"] >= B_SHARE:
        reading = "D03_B_BOUND_LIMITED"
    elif point["share_required_lt_tau"] >= A_SHARE and point["share_bad_vs_nearest"] > A_BAD:
        reading = "D03_A_CAPACITY_NOT_THE_LIMIT"
    else:
        reading = "D03_NEITHER"
    c_flag = point["share_mp_required_lt_tau"] >= C_SHARE and point["v22a_gate_hit@1_low"] > 0 and point["v22a_gate_full_coverage@5_high"] < 0
    analysed = np.zeros(z["row"].size, dtype=bool)
    analysed[Q0["q"]] = True
    flows = {"analysed_queries": gold_flows(z, analysed), "queries_with_ge2_in_pool_golds": gold_flows(z, z["gold_in_pool"] >= 2),
             "all_queries": gold_flows(z, np.ones(z["row"].size, dtype=bool))}
    pool = {"queries": int(z["row"].size), "mean_abs_delta": round(float(z["q/delta_abs_mean"].mean()), 4),
            "max_abs_delta_per_query": dist(z["q/delta_abs_max"]), "share_candidates_abs_delta_ge_0.9": round(float(z["q/delta_sat_share"].mean()), 4)}
    meta = json.loads((OUT / "score_meta.json").read_text(encoding="utf-8"))
    rec = {"utc": utc(), "phase": "UMLP_D03_SCORE_GAP_FEASIBILITY", "population": "2wiki V2_GATE", "queries": int(z["row"].size),
           "held_half_read": False, "models_fitted": 0, "units": {"s_hat": "within-query z of the twin score (v2.2A's s0_hat)", "tau": TAU, "epsilon": EPS},
           "integrity": {"score_pass": meta, "groups_equal_d01": True, "sidecar_sha256": sha256_file(SIDECAR)},
           "reading": reading, "d03_c_local_probe_not_global": bool(c_flag),
           "reading_inputs": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in point.items()},
           "thresholds": {"D03_B": f"share(required_lift >= 2 tau) >= {B_SHARE}",
                          "D03_A": f"share(required_lift < tau) >= {A_SHARE:.4f} and share(bad vs nearest) > {A_BAD}",
                          "D03_C": f"share(G_miss_mp required_lift < tau) >= {C_SHARE:.4f} and gate hit@1 low > 0 and full_coverage@5 high < 0"},
           "per_seed": per_seed, "gold_flows_descriptive": flows, "v22a_pool_offset": pool}
    RECORD.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    TABLES.write_text(tables(rec), encoding="utf-8")
    frozen_unchanged()
    log(f"read: reading {reading}; D03_C flag {bool(c_flag)}; {json.dumps(rec['reading_inputs'])}")
    return rec


# ── tables for the document ──────────────────────────────────────────────────


def fs(s: dict) -> str:
    return "n/a" if s["share"] is None else f"{s['share']:.3f} [{s['low']:.3f}, {s['high']:.3f}]"


def fd(d: dict) -> str:
    return "n/a" if d["n"] == 0 else f"{d['median']:.2f} ({d['q25']:.2f} to {d['q75']:.2f})"


def ft(d: dict) -> str:
    return "n/a" if d["n"] == 0 else f"{d['q10']:.2f} / {d['q90']:.2f} / {d['q95']:.2f}"


def fq(d: dict, k: str = "median", fmt: str = ".2f") -> str:
    return "n/a" if d["n"] == 0 else format(d[k], fmt)


def tables(rec: dict) -> str:
    L = []
    s0 = rec["per_seed"][TWIN]["strata"]
    L += ["## capacity (twin s0 missed golds; s_hat units)", "",
          "| stratum | units | queries | required lift median (q25 to q75) | q10 / q90 / q95 | < tau (one-sided) | < 2 tau (two-sided) | >= 2 tau | rank-5 is gold |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, b in s0.items():
        u = b["units"]
        L.append(f"| {s} | {u['units']} | {u['queries']} | {fd(u['required_lift'])} | {ft(u['required_lift'])} | {fs(u['one_sided_lt_tau'])} | "
                 f"{fs(u['two_sided_lt_2tau'])} | {fs(u['beyond_2tau'])} | {fs(u['rank5_is_gold'])} |")
    L += ["", "## displacers (twin s0, all strata rows)", "",
          "| stratum | displacer | pairs | required r median (q25 to q75) | q10 / q90 / q95 | r < tau | r < 2 tau |", "|---|---|---|---|---|---|---|"]
    for s, b in s0.items():
        for d in ("nearest", "hardest"):
            x = b["units"][f"{d}_displacer"]
            L.append(f"| {s} | {d} | {x['required']['n']} | {fd(x['required'])} | {ft(x['required'])} | {fs(x['one_sided_lt_tau'])} | {fs(x['two_sided_lt_2tau'])} |")
        x = b["pairs_all_displacers"]
        L.append(f"| {s} | all pairs | {x['pairs']} | {fd(x['required'])} | {ft(x['required'])} | {fs(x['one_sided_lt_tau'])} | {fs(x['two_sided_lt_2tau'])} |")
    L += ["", "## learned v2.2A movement m = delta(g) - delta(n)", "",
          "| stratum | displacer | correct (m>0) | wrong (m<0) | none | reversed | far too small | bad (m<=0 or far too small) | m median (q25 to q75) | m - r median | m / r median |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, b in s0.items():
        for d in ("nearest", "hardest"):
            x = b["units"][f"{d}_displacer"]["learned"]
            L.append(f"| {s} | {d} | {fs(x['correct'])} | {fs(x['wrong'])} | {fs(x['none'])} | {fs(x['reversed'])} | {fs(x['far_too_small'])} | {fs(x['bad'])} | "
                     f"{fd(x['m'])} | {fq(x['m_minus_r'])} | {fq(x['m_over_r'])} |")
        x = b["pairs_all_displacers"]["learned"]
        L.append(f"| {s} | all pairs | {fs(x['correct'])} | {fs(x['wrong'])} | {fs(x['none'])} | {fs(x['reversed'])} | {fs(x['far_too_small'])} | {fs(x['bad'])} | "
                 f"{fd(x['m'])} | | |")
    L += ["", "## the gold's own offset and realised entry (twin s0 missed golds)", "",
          "| stratum | delta(g) median (q25 to q75) | delta(g) q10 / q90 / q95 | abs delta(g) >= 0.9 | delta(nearest n) median | realised entry | entry if < tau | entry if >= tau | v2.2A rank median |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, b in s0.items():
        x = b["units"]["learned_gold"]
        dn = b["units"]["nearest_displacer"]["learned"]["delta_displacer"]
        L.append(f"| {s} | {fd(x['delta_g'])} | {ft(x['delta_g'])} | {fs(x['delta_g_saturated'])} | {fd(dn)} | {fs(x['realised_entry'])} | "
                 f"{fs(x['realised_entry_if_one_sided'])} | {fs(x['realised_entry_if_beyond_one_sided'])} | {fq(x['v22a_rank'], fmt='.0f')} |")
    L += ["", "## query-level full coverage@5 (twin s0 analysed queries)", "",
          "| stratum | queries | reachable by reranking | unreachable | required_fc median (q25 to q75) | q10 / q90 / q95 | < tau | < 2 tau | v2.2A reaches FC@5 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, b in s0.items():
        if "queries_full_coverage" not in b:
            continue
        x = b["queries_full_coverage"]
        L.append(f"| {s} | {x['queries']} | {x['reachable_by_reranking']} | {x['unreachable_by_any_reranker']} | {fd(x['required_fc'])} | {ft(x['required_fc'])} | "
                 f"{fs(x['one_sided_lt_tau'])} | {fs(x['two_sided_lt_2tau'])} | {fs(x['v22a_full_coverage@5'])} |")
    L += ["", "## seeds (capacity only; seeds 1-2 descriptive)", "",
          "| twin | stratum | units | queries | required lift median (q25 to q75) | q10 / q90 / q95 | < tau | < 2 tau | >= 2 tau |", "|---|---|---|---|---|---|---|---|---|"]
    for twin, seed in rec["per_seed"].items():
        for s in ("all", "G_miss_mp", "G_miss_both", "gold_count=2", "gold_count>=3"):
            u = seed["strata"][s]["units"]
            L.append(f"| {twin} | {s} | {u['units']} | {u['queries']} | {fd(u['required_lift'])} | {ft(u['required_lift'])} | {fs(u['one_sided_lt_tau'])} | "
                     f"{fs(u['two_sided_lt_2tau'])} | {fs(u['beyond_2tau'])} |")
    L += ["", "## gold flows twin s0 -> v2.2A (descriptive)", "", "| population | " + " | ".join(rec["gold_flows_descriptive"]["all_queries"]) + " |",
          "|---|" + "---|" * len(rec["gold_flows_descriptive"]["all_queries"])]
    for name, f in rec["gold_flows_descriptive"].items():
        L.append(f"| {name} | " + " | ".join(str(v) for v in f.values()) + " |")
    return LF.join(L) + LF


# ── stage: file ──────────────────────────────────────────────────────────────


def stage_file(date: str, log=print) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_umlp_d03_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    s0 = rec["per_seed"][TWIN]
    run = {"utc": utc(), "held_half_read": False, "models_fitted": 0, "queries": rec["queries"],
           "analysed_queries": s0["analysed_queries"], "missed_golds": s0["missed_golds"], "pairs": s0["pairs"],
           "reading": rec["reading"], "d03_c_local_probe_not_global": rec["d03_c_local_probe_not_global"], "reading_inputs": rec["reading_inputs"],
           "integrity": {"metric_mismatches": 0, "s0_hat_max_abs_diff": rec["integrity"]["score_pass"]["s0_hat_max_abs_diff"],
                         "gold_rank_mismatch_vs_d01": rec["integrity"]["score_pass"]["gold_rank_mismatch_vs_d01"], "groups_equal_d01": True},
           "sidecar_sha256": sha256_file(SIDECAR), "record_sha256": sha256_file(RECORD),
           "document": str(DOC.relative_to(ROOT)).replace(chr(92), "/"), "status_moves": "DECLARED_NOT_RUN -> RUN"}
    block = yaml.safe_dump({key: run}, sort_keys=False, width=160)
    text = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + block, encoding="utf-8")
    log(f"filed {key}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "read", "file"], required=True)
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"_run_{a.stage}.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + LF)

    decl = load_declaration()
    if a.stage == "score":
        stage_score(decl, log=log)
    elif a.stage == "read":
        stage_read(decl, log=log)
    else:
        stage_file(a.date, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
