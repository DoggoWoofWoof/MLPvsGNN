"""UMLP-D0.1: what compiled evidence the golds the twin leaves out of its top 5 carry (configs/umlp_d01_2wiki_feature_readout.yaml).

    python scripts/umlp_d01_feature_readout.py --stage score   # V2_GATE compile + inference of the six frozen checkpoints, integrity-checked
    python scripts/umlp_d01_feature_readout.py --stage read    # groups, within-query pair-AUCs, readings -> record.json
    python scripts/umlp_d01_feature_readout.py --stage doc
    python scripts/umlp_d01_feature_readout.py --stage file --date 2026_09_29

Held rows are removed before compilation. No parameter is updated (eval mode, no_grad). No M3B checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "umlp_d01_2wiki_feature_readout.yaml"
FROZEN_CONFIGS = [ROOT / "configs" / "universal_v2.yaml", ROOT / "configs" / "umlp_d0_2wiki_diagnostics.yaml"]
OUT = ROOT / "outputs" / "umlp_d01"
SIDECAR = OUT / "candidates_2wiki_gate.npz"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "UMLP_D01_2WIKI_FEATURE_READOUT.md"
LF = chr(10)

RESAMPLES, SEED, LEVEL = 1000, 0, 95
TOPK_NEG = 10
CHECKED_METRICS = ("recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "mrr", "ndcg@5", "ndcg@20", "full_coverage@5",
                   "full_coverage@20", "first_gold_rank", "gold_in_pool", "gold_total", "pool_size")
STRATA = ("all", "gold_count=2", "gold_count>=3", "h1", "h2", "h3", "none")
COMPARISONS = {"C1": ("G_miss", "G_rec"), "C2": ("G_miss", "N_disp"), "C3": ("G_miss", "N_hard"), "C4": ("G_miss_mp", "G_miss_both")}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def verify_pins(decl: dict) -> None:
    inp = decl["inputs"]
    for key, item in inp.items():
        if isinstance(item, dict) and "sha256" in item and sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise SystemExit(f"{item['path']}: not the pinned sha256; hard stop")
    for key, digest in inp["checkpoints"].items():
        if sha256_file(ROOT / inp["checkpoint_dir"] / f"{key}.pt") != digest:
            raise SystemExit(f"{key}.pt: not the pinned sha256; hard stop")


# ── stage: score ─────────────────────────────────────────────────────────────


def stored_gate_metrics(decl: dict, keys: list[str]) -> tuple[np.ndarray, dict]:
    """The stored per-query metrics of the declared checkpoints, V2_GATE rows only (held rows dropped as read)."""
    inp = decl["inputs"]
    with np.load(ROOT / inp["v2_eval_seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    out = {}
    for f in (inp["v2_eval_seed0"]["path"], inp["v2_eval_seeds_1_2"]["path"]):
        with np.load(ROOT / f) as z:
            for k in keys:
                for m in CHECKED_METRICS:
                    col = f"{k}/{m}"
                    if col in z.files:
                        out[col] = z[col][half]
    missing = [f"{k}/{m}" for k in keys for m in CHECKED_METRICS if f"{k}/{m}" not in out]
    if missing:
        raise SystemExit(f"stored metrics missing: {missing[:4]}; refusing")
    return half, out


def stage_score(decl: dict, log=print) -> None:
    if SIDECAR.exists():
        log(f"{SIDECAR} exists; not rescored")
        return
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import universal_v2_run as U   # frozen runner, imported not edited (sets the BLAS thread env)
    torch.set_num_threads(6)       # the stored eval records' "threads"
    verify_pins(decl)
    before = {p: sha256_file(p) for p in FROZEN_CONFIGS}
    cfg, cfg_m3b, cfg_h = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    selection = U.read_json(ROOT / decl["inputs"]["selection"]["path"])
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    m3b_contract = U.M3B_RUN.load_script("m3b_contract")
    name = "2wiki"
    contexts, handles, pkg, bank = U.open_contexts_v2(cfg_m3b, [name], m3b_compile)
    context, ds = contexts[name], handles[name]
    twin_keys = decl["arms_scored"]["twin"]
    gnn_keys = decl["arms_scored"]["mp_reference"]
    models = {}
    for key in twin_keys + gnn_keys:
        rec = U.read_json(ROOT / decl["inputs"]["checkpoint_dir"] / f"{key}.json")
        m = U.make_model(rec["arm"], inputs, bank, selected_gnn=selection["gnn"]["arm"])
        m.load_state_dict(torch.load(ROOT / decl["inputs"]["checkpoint_dir"] / f"{key}.pt", map_location="cpu"))
        m.eval()
        models[key] = m
    half_stored, stored = stored_gate_metrics(decl, list(models))
    m3a = pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    split = cfg_m3b["populations"]["eval_splits"][name]
    declared = cfg["m3b_incumbents"]["eval_populations_reused_here"][name]
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "eval", cfg_m3b, cfg_h, m3a, positions)
    del positions
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]):
        raise SystemExit("not the M3B eval population; refusing")
    half = U.half_labels(name, ds, split, pop.ids)
    if not np.array_equal(half, half_stored):
        raise SystemExit("recomputed halves differ from the stored halves; refusing")
    gi = np.flatnonzero(half)                 # held queries leave here, before anything is compiled or scored
    pop.ids, pop.idx, pop.golds = [pop.ids[i] for i in gi], pop.idx[gi], [pop.golds[i] for i in gi]
    with np.load(ROOT / decl["inputs"]["d0_masks"]["path"]) as z:
        d0_rows, d0_first = z["row"], {v: z[f"first_support_{v}"] for v in ("STRUCT", "FULL")}
    if not np.array_equal(d0_rows, gi):
        raise SystemExit("D0 mask rows are not these V2_GATE rows; refusing")
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    n = len(gi)
    sizes = np.asarray([p.size for p in prep.pools])
    chunk = max(1, int(24000 // max(sizes.mean(), 1)))
    columns = inputs["column_indices"]
    all_names = sorted(U.IDX, key=U.IDX.get)
    rows_q, rows_local, rows_gold, X_all = [], [], [], []
    score_cols = {f"{k}/score": [] for k in models}
    score_cols.update({f"{k}/base": [] for k in twin_keys})
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, chunk):
            idx = np.arange(start, min(start + chunk, n))
            qds, gold_locals, scal = [], [], []
            for j in idx:
                E = context.nodes.read(prep.pools[j])
                inp = U.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
                compiled = U.compile_query_v2(inp, prep.pools[j], prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
                gl = m3b_compile.gold_local_of(prep.pools[j], pop.golds[j])
                for v in ("STRUCT", "FULL"):
                    first = -1 if gl.size == 0 else next((t for t in (1, 2, 3) if float(compiled.scalars[gl, U.IDX[f"has_h{t}_{v}"]].max()) > 0), 0)
                    if first != d0_first[v][j]:
                        raise SystemExit(f"row {gi[j]}: first-support {v} {first} != D0 {d0_first[v][j]}; hard stop")
                qds.append({"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[j],
                            "seeds": compiled.seeds_local, "gold": gl, "gold_total": int(pop.golds[j].size), "emb": E})
                gold_locals.append(gl)
                scal.append(compiled.scalars)
            batch = U.pack_queries_v2(qds, context)
            ptr = batch.qptr.numpy()
            chunk_scores = {}
            for key, model in models.items():
                view = U.arm_view(model, batch, inputs)
                s_t = model(view)
                chunk_scores[key] = s_t.cpu().numpy().astype(np.float64)
                if key in twin_keys:
                    _, base_z = model.input(view)
                    chunk_scores[f"{key}/base"] = (model.readout.base_weight * base_z).cpu().numpy().astype(np.float64)
                for jj, j in enumerate(idx):
                    r = U.rank_metrics(chunk_scores[key][ptr[jj]:ptr[jj + 1]], gold_locals[jj], int(pop.golds[j].size))
                    for m in CHECKED_METRICS:
                        if r[m] != stored[f"{key}/{m}"][j]:
                            raise SystemExit(f"query {pop.ids[j]} (row {gi[j]}), {key} {m}: forward pass {r[m]} != stored "
                                             f"{stored[f'{key}/{m}'][j]}; hard stop")
            for jj, j in enumerate(idx):
                a, b = ptr[jj], ptr[jj + 1]
                keep = np.zeros(b - a, dtype=bool)
                keep[gold_locals[jj]] = True
                for key in twin_keys:
                    keep[np.argsort(-chunk_scores[key][a:b], kind="stable")[:TOPK_NEG]] = True
                loc = np.flatnonzero(keep)
                rows_q.append(np.full(loc.size, j))
                rows_local.append(loc)
                g = np.zeros(loc.size, dtype=bool)
                g[np.isin(loc, gold_locals[jj])] = True
                rows_gold.append(g)
                X_all.append(scal[jj][loc].astype(np.float32))
                for key in models:
                    score_cols[f"{key}/score"].append(chunk_scores[key][a:b][loc])
                for key in twin_keys:
                    score_cols[f"{key}/base"].append(chunk_scores[f"{key}/base"][a:b][loc])
            # full-pool ranks are what the groups need: keep each checkpoint's rank of every kept candidate
            for key in models:
                for jj, j in enumerate(idx):
                    a, b = ptr[jj], ptr[jj + 1]
                    order = np.argsort(-chunk_scores[key][a:b], kind="stable")
                    rank = np.empty(b - a, dtype=np.int64)
                    rank[order] = np.arange(1, b - a + 1)
                    score_cols.setdefault(f"{key}/rank", []).append(rank[rows_local[len(rows_local) - len(idx) + jj]])
            if (start // chunk) % 5 == 0:
                log(f"   {min(start + chunk, n)}/{n} queries compiled and scored, {time.time() - t0:.0f}s, integrity equal so far")
    for p, h in before.items():
        if sha256_file(p) != h:
            raise SystemExit(f"{p.name} changed during the run; hard stop")
    out = {"query": np.concatenate(rows_q), "local": np.concatenate(rows_local), "is_gold": np.concatenate(rows_gold),
           "X": np.concatenate(X_all), "row": gi.astype(np.int64), "gold_total": np.asarray([g.size for g in pop.golds]),
           "pool_size": sizes, "first_support_STRUCT": d0_first["STRUCT"], "first_support_FULL": d0_first["FULL"],
           "column_names": np.asarray(all_names), "read_columns": np.asarray(inputs["columns"])}
    out.update({k: np.concatenate(v) for k, v in score_cols.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = SIDECAR.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(SIDECAR)
    meta = {"utc": utc(), "queries": n, "candidates_kept": int(out["query"].size), "seconds": round(time.time() - t0, 1),
            "chunk_queries": chunk, "threads": 6, "mismatches": 0,
            "integrity": "rank_metrics of every checkpoint's forward pass equal the stored per-query metrics on every V2_GATE query; first-support buckets equal D0",
            "sha256": sha256_file(SIDECAR)}
    (OUT / "score_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"score: {n} queries, {meta['candidates_kept']} candidates kept, 0 mismatches, {meta['seconds']}s")


# ── the statistics ───────────────────────────────────────────────────────────


def pair_auc_query(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """P(x > y) + 0.5 P(x == y) over all (x, y) pairs of one query, per column."""
    gt = (X[:, None, :] > Y[None, :, :]).mean(axis=(0, 1))
    eq = (X[:, None, :] == Y[None, :, :]).mean(axis=(0, 1))
    return gt + 0.5 * eq


def bootstrap_mean(A: np.ndarray) -> dict:
    """A: queries x columns of per-query AUCs (queries with pairs only). Query-level percentile bootstrap, pilot rule,
    via resample-count weights so each resample is one matrix product."""
    q = A.shape[0]
    if q == 0:
        return {"mean": np.full(A.shape[1], np.nan), "low": np.full(A.shape[1], np.nan), "high": np.full(A.shape[1], np.nan), "n": 0}
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(q, size=q), minlength=q) for _ in range(RESAMPLES)]).astype(np.float64)
    means = (W @ A) / q
    lo, hi = np.percentile(means, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2], axis=0)
    return {"mean": A.mean(axis=0), "low": lo, "high": hi, "n": q}


def groups_for_seed(z: dict, twin: str, gnn_keys: list[str]) -> dict[str, np.ndarray]:
    """Per kept candidate, the declared group membership for one twin seed (boolean masks over the kept rows), plus
    the analysed-query mask (twin top-1 is a gold)."""
    rank = z[f"{twin}/rank"]
    gold = z["is_gold"]
    qi = z["query"]
    n_q = z["row"].size
    top_gold = np.zeros(n_q, dtype=bool)
    top_gold[qi[gold & (rank == 1)]] = True
    analysed = top_gold[qi]
    gnn_top5 = np.sum([z[f"{k}/rank"] <= 5 for k in gnn_keys], axis=0)
    g = {"G_top": analysed & gold & (rank == 1), "G_rec": analysed & gold & (rank >= 2) & (rank <= 5),
         "G_miss": analysed & gold & (rank > 5)}
    g["G_miss_mp"] = g["G_miss"] & (gnn_top5 >= 2)
    g["G_miss_both"] = g["G_miss"] & (gnn_top5 <= 1)
    g["N_hard"] = analysed & ~gold & (rank >= 2) & (rank <= TOPK_NEG)
    g["N_disp"] = analysed & ~gold & (rank >= 2) & (rank <= 5)
    return g, top_gold


def strata_masks(z: dict) -> dict[str, np.ndarray]:
    gt, fs = z["gold_total"], z["first_support_STRUCT"]
    return {"all": np.ones(gt.size, dtype=bool), "gold_count=2": gt == 2, "gold_count>=3": gt >= 3,
            "h1": fs == 1, "h2": fs == 2, "h3": fs == 3, "none": fs == 0}


def compare(V: np.ndarray, qi: np.ndarray, gx: np.ndarray, gy: np.ndarray, qmask: np.ndarray) -> dict:
    """Per-query pair-AUC of X-group vs Y-group candidates, over the queries in qmask that have both groups."""
    order = np.argsort(qi, kind="stable")
    qs, starts = np.unique(qi[order], return_index=True)
    ends = np.append(starts[1:], order.size)
    rows = []
    for q, a, b in zip(qs, starts, ends):
        if not qmask[q]:
            continue
        idx = order[a:b]
        xi, yi = idx[gx[idx]], idx[gy[idx]]
        if xi.size and yi.size:
            rows.append(pair_auc_query(V[xi], V[yi]))
    A = np.asarray(rows) if rows else np.zeros((0, V.shape[1]))
    return bootstrap_mean(A)


# ── stage: read ──────────────────────────────────────────────────────────────


def column_tiers(decl: dict, z: dict) -> dict:
    names = [str(s) for s in z["column_names"]]
    read = set(str(s) for s in z["read_columns"])
    screen = json.loads((ROOT / "outputs" / "universal_v2" / "feature_screen.json").read_text(encoding="utf-8"))
    unread = set(screen["dropped"])
    fams = decl["readout_columns"]["families_for_readings"]
    local_globs = fams["structural_local"]
    deep_globs = [g.replace("h1", "h2") for g in local_globs if "h1" in g] + [g.replace("h1", "h3") for g in local_globs if "h1" in g]
    deep_globs += ["branch_*", "opath_*", "typed_walks_*", "relpath_*", "relchain2_*"]
    structural = {n for n in names if any(fnmatch.fnmatch(n, g) for g in local_globs + deep_globs)}
    return {"names": names, "read": read, "unread": unread, "structural": structural}


def diagnose(decl: dict, z: dict) -> dict:
    twin_keys = decl["arms_scored"]["twin"]
    gnn_keys = decl["arms_scored"]["mp_reference"]
    tiers = column_tiers(decl, z)
    names = tiers["names"]
    qi = z["query"]
    gnn_rank = np.mean([z[f"{k}/rank"] for k in gnn_keys], axis=0)
    strata = strata_masks(z)
    per_seed = {}
    for twin in twin_keys:
        g, top_gold = groups_for_seed(z, twin, gnn_keys)
        score_terms = np.stack([z[f"{twin}/score"], z[f"{twin}/base"], z[f"{twin}/score"] - z[f"{twin}/base"], -gnn_rank], axis=1)
        V = np.concatenate([z["X"].astype(np.float64), score_terms], axis=1)
        cols = names + ["twin_score", "twin_base", "twin_residual", "gnn_score"]
        seed = {"analysed_queries": int(top_gold.sum()),
                "group_sizes": {k: int(v.sum()) for k, v in g.items()},
                "comparisons": {}}
        for cname, (x, y) in COMPARISONS.items():
            seed["comparisons"][cname] = {}
            for sname, smask in strata.items():
                res = compare(V, qi, g[x], g[y], smask & top_gold)
                seed["comparisons"][cname][sname] = {
                    "queries": res["n"],
                    "columns": {c: {"auc": _r(res["mean"][i]), "low": _r(res["low"][i]), "high": _r(res["high"][i])} for i, c in enumerate(cols)}}
        per_seed[twin] = seed
    readings = read_off(per_seed, tiers)
    return {"columns": {"read": sorted(tiers["read"]), "unread": sorted(tiers["unread"]), "structural": sorted(tiers["structural"])},
            "per_seed": per_seed, "readings": readings}


def _r(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def read_off(per_seed: dict, tiers: dict) -> dict:
    seeds = list(per_seed)
    read_struct = sorted(tiers["read"] & tiers["structural"])
    unread = sorted(tiers["unread"])

    def c2(seed, col):
        return per_seed[seed]["comparisons"]["C2"]["h1"]["columns"][col]

    def above(cell, bar):
        return cell["low"] is not None and cell["low"] > bar

    def below(cell, bar):
        return cell["high"] is not None and cell["high"] < bar

    sep_060 = {s: [c for c in read_struct if above(c2(s, c), 0.60)] for s in seeds}
    sep_055_any = any(above(c2(s, c), 0.55) for s in seeds for c in read_struct)
    resid_down = {s: below(c2(s, "twin_residual"), 0.50) for s in seeds}
    r1 = ("SUPPORTED" if all(sep_060[s] and resid_down[s] for s in seeds)
          else "NOT_SUPPORTED" if not sep_055_any else "INCONCLUSIVE")
    all_cols = sorted(tiers["read"] | tiers["unread"])
    none_055 = {s: not any(above(c2(s, c), 0.55) for c in all_cols) for s in seeds}
    gnn_060 = {s: above(c2(s, "gnn_score"), 0.60) for s in seeds}
    read_all3 = [c for c in sorted(tiers["read"]) if all(above(c2(s, c), 0.60) for s in seeds)]
    r2 = ("SUPPORTED" if all(none_055[s] and gnn_060[s] for s in seeds)
          else "NOT_SUPPORTED" if read_all3 else "INCONCLUSIVE")
    screen_partner = {}
    try:
        screen = json.loads((ROOT / "outputs" / "universal_v2" / "feature_screen.json").read_text(encoding="utf-8"))
        best = {}
        for pair in screen.get("duplicate_pairs", []) or []:   # the kept partner = the read earlier column of highest |spearman|
            later, earlier, rho = pair["later"], pair["earlier"], abs(float(pair["abs_spearman"]))
            if earlier in tiers["read"] and rho > best.get(later, -1.0):
                best[later] = rho
                screen_partner[later] = earlier
    except (OSError, ValueError):
        pass
    r3_cols = []
    for c in unread:
        if all(above(c2(s, c), 0.60) for s in seeds):
            partner = screen_partner.get(c)
            if partner is None or any(not above(c2(s, partner), 0.55) for s in seeds):
                r3_cols.append({"column": c, "kept_partner": partner})
    return {
        "R1_visible_but_scored_down": {"verdict": r1, "read_structural_columns_C2_above_0.60_h1": sep_060,
                                       "twin_residual_C2_below_0.50_h1": resid_down, "any_read_structural_above_0.55": sep_055_any},
        "R2_invisible_in_the_compiled_basis": {"verdict": r2, "no_column_above_0.55_h1": none_055, "gnn_score_C2_above_0.60_h1": gnn_060,
                                               "read_columns_above_0.60_on_all_three_seeds": read_all3},
        "R3_compiled_but_unread": {"verdict": "SUPPORTED" if r3_cols else "NOT_SUPPORTED", "columns": r3_cols,
                                   "partner_map_found": bool(screen_partner)},
    }


def stage_read(decl: dict, log=print) -> None:
    if RECORD.exists():
        raise SystemExit(f"{RECORD} exists: read once")
    meta = json.loads((OUT / "score_meta.json").read_text(encoding="utf-8"))
    if sha256_file(SIDECAR) != meta["sha256"] or meta["mismatches"] != 0:
        raise SystemExit("the sidecar is not the one the integrity-checked score stage wrote; refusing")
    before = {p: sha256_file(p) for p in FROZEN_CONFIGS}
    with np.load(SIDECAR) as f:
        z = {k: f[k] for k in f.files}
    t0 = time.time()
    result = diagnose(decl, z)
    for p, h in before.items():
        if sha256_file(p) != h:
            raise SystemExit(f"{p.name} changed; hard stop")
    record = {"phase": decl["phase"], "utc": utc(), "population": "2wiki dev V2_GATE, twin top-1 gold, per twin seed", "held_half_read": False,
              "mp_reference": decl["arms_scored"]["mp_reference"], "gat_loaded": False,
              "gat_note": "the user's go-ahead ('run') did not rule on an M3B GAT inference pass; run with the declared default u_gnn_v2_ef",
              "score": meta, "frozen_config_sha256": {p.name: h for p, h in before.items()}, "seconds": round(time.time() - t0, 1), **result}
    RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")
    log(f"record {RECORD} sha256 {sha256_file(RECORD)[:12]}")
    for k, v in result["readings"].items():
        log(f"   {k}: {v['verdict']}")


# ── stages: doc, file ────────────────────────────────────────────────────────


def fmt(cell) -> str:
    return "-" if cell["auc"] is None else f"{cell['auc']:.3f} [{cell['low']:.3f}, {cell['high']:.3f}]"


def render_doc(rec: dict) -> str:
    seeds = list(rec["per_seed"])
    s0 = seeds[0]
    r = rec["readings"]
    L = ["# UMLP-D0.1: what the golds the twin misses look like (2wiki)", "",
         "Analysis only (configs/umlp_d01_2wiki_feature_readout.yaml). 2wiki dev **V2_GATE**; queries where the twin's rank-1 item is a gold, "
         "formed per twin seed. Inference-only forward pass of the frozen twin (u_mlp_v2_mix s0-2) and GNN (u_gnn_v2_ef s0-2) checkpoints; "
         f"every stored per-query metric reproduced exactly ({rec['score']['queries']:,} queries, {rec['score']['mismatches']} mismatches). "
         "No fit, no new column, no held row. The M3B GAT was not loaded; 'MP recovers' means u_gnn_v2_ef (top 5 on >= 2 of 3 seeds).", "",
         "Statistic: within-query pair-AUC, P(x > y) + 0.5 P(x = y) over pairs from the same query, averaged per query then over queries; "
         "query bootstrap, 1000 resamples, 95 %. 0.5 = indistinguishable; above 0.5 = the first group has larger values.", "",
         "## Readings", "", "| reading | verdict |", "|---|---|"]
    for k, v in r.items():
        L.append(f"| {k} | **{v['verdict']}** |")
    L += ["", f"R1 evidence (h1 stratum, C2 = missed gold vs displacing non-gold): read structural columns with CI above 0.60 per seed: "
          + "; ".join(f"{s.split('__')[-1]} {len(v)}" for s, v in r["R1_visible_but_scored_down"]["read_structural_columns_C2_above_0.60_h1"].items())
          + f"; twin residual below 0.50 per seed: {r['R1_visible_but_scored_down']['twin_residual_C2_below_0.50_h1']}.",
          f"R2 evidence: GNN score C2 above 0.60 per seed: {r['R2_invisible_in_the_compiled_basis']['gnn_score_C2_above_0.60_h1']}; "
          f"read columns above 0.60 on all three seeds: {len(r['R2_invisible_in_the_compiled_basis']['read_columns_above_0.60_on_all_three_seeds'])}.",
          f"R3 columns: {[c['column'] for c in r['R3_compiled_but_unread']['columns']] or 'none'}.", "",
          "## Group sizes", "", "| twin seed | analysed queries | " + " | ".join(rec["per_seed"][s0]["group_sizes"]) + " |",
          "|---|---:|" + "---:|" * len(rec["per_seed"][s0]["group_sizes"])]
    for s in seeds:
        v = rec["per_seed"][s]
        L.append(f"| {s.split('__')[-1]} | {v['analysed_queries']:,} | " + " | ".join(f"{x:,}" for x in v["group_sizes"].values()) + " |")
    for cname, desc in (("C2", "missed gold vs displacing non-gold (ranks 2-5)"), ("C1", "missed gold vs recovered extra gold"),
                        ("C4", "missed golds the GNN recovers vs missed by both")):
        L += ["", f"## {cname}: {desc} -- h1 stratum, seed 0, score terms and the 20 columns furthest from 0.5", ""]
        cell = rec["per_seed"][s0]["comparisons"][cname]["h1"]
        L.append(f"Queries with both groups: {cell['queries']:,}. Seeds 1-2 in the record.")
        L += ["", "| column | tier | AUC [95% CI] s0 | s1 | s2 |", "|---|---|---|---|---|"]
        cols = cell["columns"]
        ranked = sorted((c for c in cols if cols[c]["auc"] is not None and c not in ("twin_score", "twin_base", "twin_residual", "gnn_score")),
                        key=lambda c: -abs(cols[c]["auc"] - 0.5))[:20]
        for c in ["twin_score", "twin_base", "twin_residual", "gnn_score"] + ranked:
            tier = "score" if c in ("twin_score", "twin_base", "twin_residual", "gnn_score") else ("read" if c in rec["columns"]["read"] else "unread" if c in rec["columns"]["unread"] else "-")
            L.append(f"| {c} | {tier} | {fmt(cols[c])} | " + " | ".join(fmt(rec["per_seed"][s]["comparisons"][cname]["h1"]["columns"][c]) for s in seeds[1:]) + " |")
    L += ["", "## C2 by stratum, seed 0 (score terms)", "", "| stratum | queries | twin_base | twin_residual | gnn_score |", "|---|---:|---|---|---|"]
    for st, cell in rec["per_seed"][s0]["comparisons"]["C2"].items():
        c = cell["columns"]
        L.append(f"| {st} | {cell['queries']:,} | {fmt(c['twin_base'])} | {fmt(c['twin_residual'])} | {fmt(c['gnn_score'])} |")
    L += ["", "## Limits", "",
          "- Univariate: a column that does not separate alone may separate jointly; that needs a fit, which this file bars.",
          "- 'Evidence collapsed by aggregation' (Case C) is not decidable here: it needs a new compiled summary.",
          "- The GNN reference is u_gnn_v2_ef, not the M3B GAT (not loaded).", ""]
    return LF.join(L)


def stage_doc(log=print) -> None:
    DOC.write_text(render_doc(json.loads(RECORD.read_text(encoding="utf-8"))), encoding="utf-8")
    log(f"doc {DOC}")


def stage_file(date: str, log=print) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    key = f"run_record_umlp_d01_{date}"
    if key + ":" in text:
        raise SystemExit(f"{key} already filed")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    block = {key: {"utc": utc(), "record": RECORD.relative_to(ROOT).as_posix(), "record_sha256": sha256_file(RECORD),
                   "sidecar_sha256": rec["score"]["sha256"], "integrity_mismatches": rec["score"]["mismatches"],
                   "document": DOC.relative_to(ROOT).as_posix(), "held_half_read": False, "gat_loaded": False,
                   "go_ahead": "user 2026-09-29: 'run stop asking me for permissions all the time' -- run with the declared default MP reference",
                   "readings": {k: v["verdict"] for k, v in rec["readings"].items()},
                   "status_moves": "DECLARED_NOT_RUN -> RUN", "next": "report; v2.1 is its own dated declaration"}}
    new = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    if new == text:
        raise SystemExit("status line not found")
    CONFIG.write_text(new.rstrip(LF) + LF + LF + yaml.safe_dump(block, sort_keys=False, width=200), encoding="utf-8")
    log(f"filed {key}; status RUN")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "read", "doc", "file"], required=True)
    ap.add_argument("--date")
    args = ap.parse_args()
    decl = load_declaration()

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
    if args.stage == "score":
        stage_score(decl, log)
    elif args.stage == "read":
        stage_read(decl, log)
    elif args.stage == "doc":
        stage_doc(log)
    else:
        stage_file(args.date, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
