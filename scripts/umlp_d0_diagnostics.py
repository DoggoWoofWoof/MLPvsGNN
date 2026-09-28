"""UMLP-D0: where u_mlp_v2_mix falls short on 2wiki -- analysis only (configs/umlp_d0_2wiki_diagnostics.yaml).

    python scripts/umlp_d0_diagnostics.py --stage masks   # amendment A1: V2_GATE masks by the frozen compiler, checked
    python scripts/umlp_d0_diagnostics.py --stage read    # the slices, gaps, gate readouts and the three readings
    python scripts/umlp_d0_diagnostics.py --stage doc     # docs/UMLP_D0_2WIKI_DIAGNOSTICS.md from the record only
    python scripts/umlp_d0_diagnostics.py --stage file --date 2026_09_28   # run_record_umlp_d0_<date>, status RUN

Held rows are dropped before any metric is formed (read) and before any feature is compiled (masks). No model is
built or loaded. Every input is checked against the sha256 the declaration pins; a mismatch refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "umlp_d0_2wiki_diagnostics.yaml"
V2_CONFIG = ROOT / "configs" / "universal_v2.yaml"
OUT = ROOT / "outputs" / "umlp_d0"
MASKS = OUT / "masks_2wiki_gate.npz"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "UMLP_D0_2WIKI_DIAGNOSTICS.md"
LF = chr(10)

BLOCK_NAMES = ["STRUCT_h1", "STRUCT_h2", "STRUCT_h3", "FULL_h1", "FULL_h2", "FULL_h3", "TYPED_h1", "TYPED_h2", "TYPED_h3"]
READABLE_BLOCKS = list(range(6))            # TYPED blocks are inert on 2wiki (declaration gate_readouts)
MASK_NAMES = ([f"has_h{t}_{v}" for v in ("STRUCT", "FULL") for t in (1, 2, 3)]
              + [f"ring_n_h{t}_{v}" for v in ("STRUCT", "FULL") for t in (1, 2, 3)]
              + [f"typed_walks_h{t}" for t in (1, 2, 3)])
METRICS = ("recall@5", "full_coverage@5", "hit@1", "mrr")
RESAMPLES, SEED, LEVEL = 1000, 0, 95        # the pilot rule (universal_v2_run.BOOTSTRAP)
GATE_COUNTS = {"2wiki": 6290, "metaqa": 19738, "squad": 5841}


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


def verify_inputs(decl: dict) -> dict:
    inputs = decl["inputs"]
    pinned = {k: v for k, v in inputs.items() if isinstance(v, dict) and "sha256" in v}
    pinned.update({"ctx_" + k: v for k, v in inputs["gate_value_context"].items()})
    got = {}
    for key, item in pinned.items():
        path = ROOT / item["path"]
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise SystemExit(f"{item['path']}: sha256 {digest[:12]} is not the pinned {item['sha256'][:12]}; refusing (hard stop)")
        got[key] = {"path": item["path"], "sha256": digest}
    return got


# ── the statistics (the pilot's paired rule, reproduced) ─────────────────────


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """universal_v2_run.paired_bootstrap: mean of a - b per query, percentile interval over query resamples."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = int(d.size)
    if n == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(SEED)
    means = np.asarray([d[rng.integers(n, size=n)].mean() for _ in range(RESAMPLES)])
    lo, hi = np.percentile(means, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return {"mean": round(float(d.mean()), 4), "low": round(float(lo), 4), "high": round(float(hi), 4), "n": n}


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import rankdata
    n = int(x.size)
    if n < 3:
        return {"rho": None, "low": None, "high": None, "n": n}

    def rho(i):
        rx, ry = rankdata(x[i]), rankdata(y[i])
        sx, sy = rx.std(), ry.std()
        return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy)) if sx > 0 and sy > 0 else 0.0
    rng = np.random.default_rng(SEED)
    boots = np.asarray([rho(rng.integers(n, size=n)) for _ in range(RESAMPLES)])
    lo, hi = np.percentile(boots, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return {"rho": round(rho(np.arange(n)), 4), "low": round(float(lo), 4), "high": round(float(hi), 4), "n": n}


def below(iv: dict) -> bool:
    return iv["high"] is not None and iv["high"] < 0


def above(iv: dict) -> bool:
    return iv["low"] is not None and iv["low"] > 0


# ── loading, V2_GATE rows only ───────────────────────────────────────────────


def arm_keys(decl: dict) -> dict:
    a = decl["arms_read"]
    return {"twin": a["twin"], "twin_ungated": a["twin_ungated"], "core": a["core"], "mp": a["mp_incumbent"],
            "mp_struct": a["mp_incumbent_struct_only"], "no_mp": a["mp_no_mp_control"], "gnn": a["selected_gnn"],
            "fixed_rrf": ["fixed:rrf"]}


def load_gate_arrays(decl: dict) -> dict:
    """Every declared arm's metric columns plus the slice sources, V2_GATE rows only. The held rows are removed
    from each column as it is read; nothing downstream can see them."""
    inp = decl["inputs"]
    files = [ROOT / inp["v2_eval_seed0"]["path"], ROOT / inp["v2_eval_seeds_1_2"]["path"], ROOT / inp["m3b_eval"]["path"]]
    with np.load(files[0]) as z:
        half = z["half"].astype(bool)
    if int(half.sum()) != GATE_COUNTS["2wiki"] or half.size != 12576:
        raise SystemExit(f"2wiki halves {int(half.sum())} / {half.size}: not the filed counts; refusing")
    with np.load(files[1]) as z:
        if not np.array_equal(z["half"].astype(bool), half):
            raise SystemExit("the supplement record's halves differ from the seed-0 record; refusing")
    wanted = {}
    for group, keys in arm_keys(decl).items():
        for k in keys:
            for m in METRICS + ("gold_total", "gold_in_pool", "pool_size"):
                wanted[f"{k}/{m}"] = None
    for b in range(9):
        for s in decl["arms_read"]["twin"]:
            wanted[f"{s}/block_gate{b}"] = None
    out = {"half_n": int(half.sum())}
    for f in files:
        with np.load(f) as z:
            for key in z.files:
                if key in wanted and wanted[key] is None:
                    col = z[key]
                    if col.size != half.size:
                        raise SystemExit(f"{f.name}:{key} has {col.size} rows, not {half.size}; refusing")
                    wanted[key] = col[half]
            if f == files[0]:
                out["gold_dist_struct"] = z["gold_dist_struct"][half]
                out["pool_size"] = z["pool_size"][half]
                out["hop"] = z["hop"][half]
    missing = [k for k, v in wanted.items() if v is None and not k.endswith(("gold_total", "gold_in_pool", "pool_size"))]
    if missing:
        raise SystemExit(f"declared columns missing: {missing[:5]}; refusing")
    out["cols"] = {k: v for k, v in wanted.items() if v is not None}
    ref = out["cols"]["fixed:rrf/pool_size"]
    for k, v in out["cols"].items():
        if k.endswith("/pool_size") and not np.array_equal(v, ref):
            raise SystemExit(f"{k}: pool sizes differ from fixed:rrf query by query; refusing")
    out["gold_total"] = out["cols"]["fixed:rrf/gold_total"]
    out["gold_in_pool"] = out["cols"]["fixed:rrf/gold_in_pool"]
    return out


def load_context_gates(decl: dict) -> dict:
    """metaqa and squad block gates of the twin, V2_GATE rows; no metric column is opened."""
    out = {}
    ctx = decl["inputs"]["gate_value_context"]
    for name in ("metaqa", "squad"):
        with np.load(ROOT / ctx[f"{name}_seed0"]["path"]) as z:
            half = z["half"].astype(bool)
            if int(half.sum()) != GATE_COUNTS[name]:
                raise SystemExit(f"{name}: V2_GATE counts {int(half.sum())}, not {GATE_COUNTS[name]}; refusing")
            per_seed = {}
            s0 = decl["arms_read"]["twin"][0]
            per_seed[s0] = np.stack([z[f"{s0}/block_gate{b}"][half] for b in range(9)], axis=1)
        with np.load(ROOT / ctx[f"{name}_seeds_1_2"]["path"]) as z:
            for s in decl["arms_read"]["twin"][1:]:
                per_seed[s] = np.stack([z[f"{s}/block_gate{b}"][half] for b in range(9)], axis=1)
        out[name] = per_seed
    return out


def seed_mean(cols: dict, keys: list[str], metric: str) -> np.ndarray:
    return np.mean([cols[f"{k}/{metric}"] for k in keys], axis=0)


# ── slices ───────────────────────────────────────────────────────────────────


def build_slices(g: dict, masks: dict | None) -> dict:
    n = g["half_n"]
    dist = g["gold_dist_struct"]
    total = g["gold_total"]
    inpool = g["gold_in_pool"]
    size = g["pool_size"]
    t1, t2 = np.quantile(size, [1 / 3, 2 / 3])
    s = {
        "gold_struct_distance": {"no gold in pool": dist == -1, "0 (seed is gold)": dist == 0, "1": dist == 1, "2": dist == 2,
                                 "3": dist == 3, ">=4 or unreachable": dist >= 4},
        "gold_count": {"1": total == 1, "2": total == 2, ">=3": total >= 3},
        "gold_in_pool": {"all golds in pool": inpool >= total, "some": (inpool > 0) & (inpool < total), "none": inpool == 0},
        "hop": {f"hop {int(h)}": g["hop"] == h for h in np.unique(g["hop"])},
        "pool_size": {f"<= {t1:.0f}": size <= t1, f"({t1:.0f}, {t2:.0f}]": (size > t1) & (size <= t2), f"> {t2:.0f}": size > t2},
    }
    if masks is not None:
        for view in ("STRUCT", "FULL"):
            first = masks[f"first_support_{view}"]
            s[f"depth_block_activation_{view}"] = {"no gold in pool": first == -1, "a gold has h1 support": first == 1,
                                                   "else h2": first == 2, "else h3": first == 3, "no support at h<=3": first == 0}
    for name, parts in s.items():
        cover = np.zeros(n, dtype=int)
        for m in parts.values():
            cover += m.astype(int)
        if not (cover == 1).all():
            raise SystemExit(f"slice {name} does not partition the {n} V2_GATE queries; refusing")
    return s


# ── the reading ──────────────────────────────────────────────────────────────


def diagnose(decl: dict, g: dict, masks: dict | None, ctx_gates: dict) -> dict:
    keys = arm_keys(decl)
    cols = g["cols"]
    arms = {}
    for group, ks in keys.items():
        arms[group] = {m: seed_mean(cols, ks, m) for m in METRICS}
    arms["twin_s0"] = {m: cols[f"{keys['twin'][0]}/{m}"] for m in METRICS}
    gaps = {"GAP_TO_MP": ("twin", "mp"), "GAP_TO_CORE": ("twin", "core"), "GATING_EFFECT": ("twin_s0", "twin_ungated"),
            "PROPAGATION_EFFECT": ("mp", "no_mp"), "GNN_V2_OVER_TWIN": ("gnn", "twin")}
    whole = {"queries": g["half_n"], "arms": {a: {m: round(float(v[m].mean()), 4) for m in METRICS} for a, v in arms.items()},
             "gaps": {gname: {m: paired(arms[a][m], arms[b][m]) for m in METRICS} for gname, (a, b) in gaps.items()}}
    whole_gap_mp = {m: float((arms["twin"][m] - arms["mp"][m]).sum()) for m in METRICS}
    slices = build_slices(g, masks)
    per_slice = {}
    for sname, parts in slices.items():
        per_slice[sname] = {}
        for label, mask in parts.items():
            row = {"queries": int(mask.sum()), "share": round(float(mask.mean()), 4),
                   "arms": {a: {m: (round(float(arms[a][m][mask].mean()), 4) if mask.any() else None) for m in METRICS}
                            for a in ("twin", "core", "mp", "no_mp", "twin_ungated", "gnn")},
                   "gaps": {gname: {m: paired(arms[a][m][mask], arms[b][m][mask]) for m in METRICS}
                            for gname, (a, b) in gaps.items() if gname != "GNN_V2_OVER_TWIN"},
                   "contribution_to_GAP_TO_MP": {m: (round(float((arms["twin"][m][mask] - arms["mp"][m][mask]).sum() / whole_gap_mp[m]), 4)
                                                     if whole_gap_mp[m] != 0 else None) for m in METRICS}}
            per_slice[sname][label] = row
    # gate readouts
    twin_keys = decl["arms_read"]["twin"]
    gates_2wiki = {s: np.stack([cols[f"{s}/block_gate{b}"] for b in range(9)], axis=1) for s in twin_keys}
    all_gates = {"2wiki": gates_2wiki, **ctx_gates}
    q = [10, 25, 50, 75, 90]
    gate_stats = {}
    for ds, per_seed in all_gates.items():
        gate_stats[ds] = {}
        for b, bname in enumerate(BLOCK_NAMES):
            gate_stats[ds][bname] = {}
            for s, arr in per_seed.items():
                v = arr[:, b]
                gate_stats[ds][bname][s] = {"mean": round(float(v.mean()), 4), **{f"q{p}": round(float(np.percentile(v, p)), 4) for p in q}}
            sm = np.mean([per_seed[s][:, b] for s in per_seed], axis=0)
            gate_stats[ds][bname]["seed_mean_median"] = round(float(np.median(sm)), 4)
    corr = {}
    gap_core = arms["twin"]["recall@5"] - arms["core"]["recall@5"]
    gap_mp = arms["twin"]["recall@5"] - arms["mp"]["recall@5"]
    for b in READABLE_BLOCKS:
        sm = np.mean([gates_2wiki[s][:, b] for s in twin_keys], axis=0)
        corr[BLOCK_NAMES[b]] = {"vs_GAP_TO_CORE_recall@5": spearman(sm, gap_core), "vs_GAP_TO_MP_recall@5": spearman(sm, gap_mp)}
    readings = read_off(per_slice, gate_stats, gates_2wiki, twin_keys, masks is not None)
    return {"whole_population": whole, "slices": per_slice, "gate_values": gate_stats, "gate_correlations_2wiki": corr,
            "readings": readings}


def read_off(per_slice: dict, gate_stats: dict, gates_2wiki: dict, twin_keys: list[str], have_masks: bool) -> dict:
    # R1
    medians = {BLOCK_NAMES[b]: gate_stats["2wiki"][BLOCK_NAMES[b]]["seed_mean_median"] for b in READABLE_BLOCKS}
    near_half = all(0.35 <= v <= 0.65 for v in medians.values())
    closed_every_seed = all(gate_stats["2wiki"][BLOCK_NAMES[b]][s]["q50"] < 0.35 for b in READABLE_BLOCKS for s in twin_keys)
    target_cells = [("gold_struct_distance", ">=4 or unreachable")]
    if have_masks:
        target_cells.append(("depth_block_activation_STRUCT", "no support at h<=3"))
    neg_in_target = [f"{s}/{l}" for s, l in target_cells if below(per_slice[s][l]["gaps"]["GAP_TO_CORE"]["recall@5"])]
    neg_anywhere = [f"{s}/{l}" for s, parts in per_slice.items() for l, row in parts.items() if below(row["gaps"]["GAP_TO_CORE"]["recall@5"])]
    if near_half and neg_in_target:
        r1 = "SUPPORTED"
    elif not neg_anywhere or closed_every_seed:
        r1 = "NOT_SUPPORTED"
    else:
        r1 = "INCONCLUSIVE"
    # R2
    dist = per_slice["gold_struct_distance"]
    share = (dist["2"]["contribution_to_GAP_TO_MP"]["recall@5"] or 0.0) + (dist["3"]["contribution_to_GAP_TO_MP"]["recall@5"] or 0.0)
    prop_pos = all(above(dist[b]["gaps"]["PROPAGATION_EFFECT"]["recall@5"]) for b in ("2", "3"))
    r2 = "SUPPORTED" if (share >= 0.5 and prop_pos) else "NOT_SUPPORTED" if share < 0.25 else "INCONCLUSIVE"
    # R3 -- the multi-gold buckets together, from the per-bucket sums (weighted by queries)
    gc = per_slice["gold_count"]
    multi = [gc["2"], gc[">=3"]]
    nq = sum(r["queries"] for r in multi)

    def pooled(metric):
        return sum(r["gaps"]["GAP_TO_MP"][metric]["mean"] * r["queries"] for r in multi if r["queries"]) / nq if nq else 0.0
    fc, h1 = pooled("full_coverage@5"), pooled("hit@1")
    r3_cis = all(below(r["gaps"]["GAP_TO_MP"][m]) for r in multi if r["queries"] for m in ("full_coverage@5", "hit@1"))
    r3 = "SUPPORTED" if (r3_cis and fc <= 2 * h1 and fc < 0) else "NOT_SUPPORTED"
    return {
        "R1_residual_gating_hypothesis": {"verdict": r1, "seed_mean_median_gate_2wiki": medians, "gates_near_initialisation": near_half,
                                          "gates_closed_on_every_seed": closed_every_seed, "GAP_TO_CORE_below_zero_in_target_cells": neg_in_target,
                                          "GAP_TO_CORE_below_zero_anywhere": neg_anywhere},
        "R2_shortfall_sits_where_propagation_pays": {"verdict": r2, "share_of_GAP_TO_MP_in_distance_2_and_3": round(share, 4),
                                                     "PROPAGATION_EFFECT_above_zero_in_2_and_in_3": prop_pos,
                                                     "note": "a location, not an attribution"},
        "R3_set_coverage": {"verdict": r3, "multi_gold_queries": nq, "GAP_TO_MP_full_coverage@5": round(fc, 4), "GAP_TO_MP_hit@1": round(h1, 4),
                            "both_intervals_below_zero_in_each_multi_gold_bucket": r3_cis},
    }


# ── stage: masks (amendment A1) ──────────────────────────────────────────────


def stage_masks(decl: dict, log=print) -> None:
    if MASKS.exists():
        log(f"{MASKS} exists; not recompiled")
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    import universal_v2_run as U   # the frozen runner, imported and not edited
    cfg, cfg_m3b, cfg_h = U.load_configs()
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    m3b_contract = U.M3B_RUN.load_script("m3b_contract")
    name = "2wiki"
    contexts, handles, pkg, _bank = U.open_contexts_v2(cfg_m3b, [name], m3b_compile)
    context, ds = contexts[name], handles[name]
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
    ids_file = json.loads((ROOT / decl["inputs"]["v2_query_ids"]["path"]).read_text(encoding="utf-8"))
    ids_file = ids_file if isinstance(ids_file, list) else ids_file["query_ids"]
    if list(pop.ids) != list(ids_file):
        raise SystemExit("the recompiled population is not in the stored order; refusing")
    half = U.half_labels(name, ds, split, pop.ids)
    with np.load(ROOT / decl["inputs"]["v2_eval_seed0"]["path"]) as z:
        if not np.array_equal(z["half"].astype(bool), half):
            raise SystemExit("recomputed halves differ from the stored halves; refusing")
        stored = {k: z[k] for k in ("gold_dist_struct", "fixed:rrf/recall@5", "fixed:rrf/hit@1", "fixed:rrf/gold_in_pool",
                                    "fixed:rrf/gold_total", "fixed:rrf/pool_size")}
    gi = np.flatnonzero(half)                  # held queries leave here, before anything is compiled
    pop.ids = [pop.ids[i] for i in gi]
    pop.idx = pop.idx[gi]
    pop.golds = [pop.golds[i] for i in gi]
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    n = len(gi)
    out = {f"gold_any_{c}": np.zeros(n, dtype=np.float32) for c in MASK_NAMES}
    out.update({f"pool_mean_{c}": np.zeros(n, dtype=np.float32) for c in MASK_NAMES})
    out["first_support_STRUCT"] = np.zeros(n, dtype=np.int64)
    out["first_support_FULL"] = np.zeros(n, dtype=np.int64)
    out["row"] = gi.astype(np.int64)
    mismatches = 0
    t0 = time.time()
    rrf = U.IDX["rrf"]
    for j in range(n):
        E = context.nodes.read(prep.pools[j])
        inp = U.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
        compiled = U.compile_query_v2(inp, prep.pools[j], prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
        gold_local = m3b_compile.gold_local_of(prep.pools[j], pop.golds[j])
        row = gi[j]
        r = U.rank_metrics(compiled.scalars[:, rrf], gold_local, int(pop.golds[j].size))
        checks = [U.gold_distance_struct(compiled.scalars, gold_local) == stored["gold_dist_struct"][row],
                  r["recall@5"] == stored["fixed:rrf/recall@5"][row], r["hit@1"] == stored["fixed:rrf/hit@1"][row],
                  r["gold_in_pool"] == stored["fixed:rrf/gold_in_pool"][row], r["gold_total"] == stored["fixed:rrf/gold_total"][row],
                  r["pool_size"] == stored["fixed:rrf/pool_size"][row]]
        if not all(checks):
            mismatches += 1
            raise SystemExit(f"query {pop.ids[j]} (row {row}): the recompile disagrees with the stored arrays {checks}; hard stop")
        X = compiled.scalars
        for c in MASK_NAMES:
            col = X[:, U.IDX[c]]
            out[f"pool_mean_{c}"][j] = float(col.mean()) if col.size else 0.0
            out[f"gold_any_{c}"][j] = float(col[gold_local].max()) if gold_local.size else 0.0
        for view in ("STRUCT", "FULL"):
            if gold_local.size == 0:
                out[f"first_support_{view}"][j] = -1
                continue
            first = 0
            for t in (1, 2, 3):
                if float(X[gold_local, U.IDX[f"has_h{t}_{view}"]].max()) > 0:
                    first = t
                    break
            out[f"first_support_{view}"][j] = first
        if (j + 1) % 500 == 0:
            log(f"   {j + 1}/{n} compiled, {time.time() - t0:.0f}s, all checks equal so far")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = MASKS.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(MASKS)
    meta = {"utc": utc(), "queries": n, "mismatches": mismatches, "seconds": round(time.time() - t0, 1),
            "integrity": "gold_dist_struct and fixed rrf recall@5 / hit@1 / gold_in_pool / gold_total / pool_size equal the stored arrays on every query",
            "sha256": sha256_file(MASKS)}
    (OUT / "masks_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"masks: {n} V2_GATE queries, 0 mismatches, {meta['seconds']}s, sha256 {meta['sha256'][:12]}")


def load_masks() -> dict | None:
    if not MASKS.exists():
        return None
    with np.load(MASKS) as z:
        return {k: z[k] for k in z.files}


# ── stages: read, doc, file ──────────────────────────────────────────────────


def stage_read(decl: dict, log=print) -> None:
    if RECORD.exists():
        raise SystemExit(f"{RECORD} exists: the diagnostics are read once")
    v2_before = sha256_file(V2_CONFIG)
    got = verify_inputs(decl)
    masks = load_masks()
    if masks is None:
        raise SystemExit("no masks: run --stage masks first (amendment A1)")
    meta = json.loads((OUT / "masks_meta.json").read_text(encoding="utf-8"))
    if sha256_file(MASKS) != meta["sha256"] or meta["mismatches"] != 0:
        raise SystemExit("the masks sidecar is not the one the integrity check wrote; refusing")
    g = load_gate_arrays(decl)
    with np.load(ROOT / decl["inputs"]["v2_eval_seed0"]["path"]) as z:
        rows = np.flatnonzero(z["half"].astype(bool))
    if not np.array_equal(rows, masks["row"]):
        raise SystemExit("mask rows are not the V2_GATE rows; refusing")
    ctx = load_context_gates(decl)
    t0 = time.time()
    result = diagnose(decl, g, masks, ctx)
    if sha256_file(V2_CONFIG) != v2_before:
        raise SystemExit("configs/universal_v2.yaml changed during the run; hard stop")
    record = {"phase": decl["phase"], "utc": utc(), "population": "2wiki dev, V2_GATE only", "held_half_read": False,
              "inputs": got, "masks": {"path": MASKS.relative_to(ROOT).as_posix(), **meta},
              "universal_v2_yaml_sha256_unchanged": v2_before, "seconds": round(time.time() - t0, 1), **result}
    OUT.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")
    log(f"record: {RECORD} sha256 {sha256_file(RECORD)[:12]}")
    for k, v in result["readings"].items():
        log(f"   {k}: {v['verdict']}")


def fmt_iv(iv: dict) -> str:
    if iv.get("mean") is None:
        return "-"
    return f"{iv['mean']:+.4f} [{iv['low']:+.4f}, {iv['high']:+.4f}]"


def render_doc(rec: dict) -> str:
    L = []
    w = rec["whole_population"]
    L += ["# UMLP-D0: where u_mlp_v2_mix falls short on 2wiki", "",
          "Analysis only, post hoc (configs/umlp_d0_2wiki_diagnostics.yaml, amendment A1). 2wiki dev, **V2_GATE half only** "
          f"({w['queries']:,} queries); the held half was not read for any arm. No fit, no forward pass. Seed-mean arms over seeds 0-2 "
          "unless named `_s0`. Intervals: paired bootstrap, 1000 resamples, default_rng(0), 95 %.", "",
          "`GAP_TO_MP` = twin minus gat_universal_v1 (what the failed twin cell measures); `GAP_TO_CORE` = twin minus qls_u_sota_v1 "
          "(the M3B non-MP incumbent on the 78-column contract -- the closest filed core, not an ablation); `GATING_EFFECT` = twin s0 minus the "
          "ungated u_mlp_v2 s0; `PROPAGATION_EFFECT` = gat_universal_v1 minus gat_no_mp_v1 (M3B's own MP contrast).", "",
          "## Readings", "", "| reading | verdict | evidence |", "|---|---|---|"]
    r = rec["readings"]
    r1, r2, r3 = r["R1_residual_gating_hypothesis"], r["R2_shortfall_sits_where_propagation_pays"], r["R3_set_coverage"]
    L.append(f"| R1 residual gating | **{r1['verdict']}** | median gates on 2wiki {', '.join(f'{k} {v:.3f}' for k, v in r1['seed_mean_median_gate_2wiki'].items())}; "
             f"GAP_TO_CORE below zero in target cells: {r1['GAP_TO_CORE_below_zero_in_target_cells'] or 'none'}; anywhere: {r1['GAP_TO_CORE_below_zero_anywhere'] or 'none'} |")
    L.append(f"| R2 shortfall where propagation pays | **{r2['verdict']}** | distance 2+3 carry {r2['share_of_GAP_TO_MP_in_distance_2_and_3']:.1%} of GAP_TO_MP (recall@5); "
             f"PROPAGATION_EFFECT above zero in 2 and in 3: {r2['PROPAGATION_EFFECT_above_zero_in_2_and_in_3']} -- a location, not an attribution |")
    L.append(f"| R3 set coverage | **{r3['verdict']}** | multi-gold ({r3['multi_gold_queries']:,} q): GAP_TO_MP full_coverage@5 {r3['GAP_TO_MP_full_coverage@5']:+.4f} vs hit@1 {r3['GAP_TO_MP_hit@1']:+.4f} |")
    L += ["", "Slice concentration (the declared slices, as they fell on this population):", ""]
    for sname, parts in rec["slices"].items():
        big = max(parts.items(), key=lambda kv: kv[1]["queries"])
        empty = [lab for lab, row in parts.items() if row["queries"] == 0]
        L.append(f"- {sname}: largest bucket `{big[0]}` holds {big[1]['queries']:,} of {w['queries']:,} queries ({big[1]['share']:.1%})"
                 + (f"; empty: {', '.join(f'`{e}`' for e in empty)}" if empty else ""))
    L += ["", "## Whole V2_GATE population", "", "| arm | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
    for a, v in w["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{v[m]:.4f}" for m in METRICS) + " |")
    L += ["", "| gap | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for gname, v in w["gaps"].items():
        L.append(f"| {gname} | " + " | ".join(fmt_iv(v[m]) for m in METRICS) + " |")
    for sname, parts in rec["slices"].items():
        L += ["", f"## Slice: {sname}", "",
              "| bucket | queries | twin R@5 | core R@5 | GAT R@5 | GAP_TO_MP R@5 | share of GAP_TO_MP | GAP_TO_CORE R@5 | GATING_EFFECT R@5 | PROPAGATION_EFFECT R@5 | GAP_TO_MP FC@5 |",
              "|---|---:|---:|---:|---:|---|---:|---|---|---|---|"]
        for label, row in parts.items():
            a = row["arms"]
            f = (lambda x: "-" if x is None else f"{x:.4f}")
            c = row["contribution_to_GAP_TO_MP"]["recall@5"]
            L.append(f"| {label} | {row['queries']:,} | {f(a['twin']['recall@5'])} | {f(a['core']['recall@5'])} | {f(a['mp']['recall@5'])} | "
                     f"{fmt_iv(row['gaps']['GAP_TO_MP']['recall@5'])} | {'-' if c is None else f'{c:.1%}'} | {fmt_iv(row['gaps']['GAP_TO_CORE']['recall@5'])} | "
                     f"{fmt_iv(row['gaps']['GATING_EFFECT']['recall@5'])} | {fmt_iv(row['gaps']['PROPAGATION_EFFECT']['recall@5'])} | "
                     f"{fmt_iv(row['gaps']['GAP_TO_MP']['full_coverage@5'])} |")
    L += ["", "## Block gates of u_mlp_v2_mix (V2_GATE, median of the per-query seed mean)", "",
          "| block | 2wiki | metaqa | squad |", "|---|---:|---:|---:|"]
    for b in BLOCK_NAMES:
        L.append(f"| {b} | " + " | ".join(f"{rec['gate_values'][d][b]['seed_mean_median']:.3f}" for d in ("2wiki", "metaqa", "squad")) + " |")
    L += ["", "TYPED blocks are structurally inert on 2wiki (no typed STRUCT relation edge): their columns are zero whatever the gate.", "",
          "Spearman correlation on 2wiki between a block's per-query gate and the query's gaps (recall@5):", "",
          "| block | vs GAP_TO_CORE | vs GAP_TO_MP |", "|---|---|---|"]
    for b, v in rec["gate_correlations_2wiki"].items():
        L.append(f"| {b} | {v['vs_GAP_TO_CORE_recall@5']['rho']:+.3f} [{v['vs_GAP_TO_CORE_recall@5']['low']:+.3f}, {v['vs_GAP_TO_CORE_recall@5']['high']:+.3f}] | "
                 f"{v['vs_GAP_TO_MP_recall@5']['rho']:+.3f} [{v['vs_GAP_TO_MP_recall@5']['low']:+.3f}, {v['vs_GAP_TO_MP_recall@5']['high']:+.3f}] |")
    m = rec["masks"]
    L += ["", "## Provenance", "",
          f"- masks: {m['queries']:,} V2_GATE queries recompiled by the frozen compiler in {m['seconds']:.0f} s, {m['mismatches']} mismatches against the stored "
          f"gold_dist_struct and fixed rrf arrays (sha256 `{m['sha256'][:16]}`)",
          f"- inputs verified against the declaration's pins: {len(rec['inputs'])} files",
          f"- configs/universal_v2.yaml unchanged (sha256 `{rec['universal_v2_yaml_sha256_unchanged'][:16]}`)",
          "- nothing here selects an architecture, a column or a threshold; a Universal-MLP-v2.1 declaration may cite it as motivation only", ""]
    return LF.join(L)


def stage_doc(log=print) -> None:
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    DOC.write_text(render_doc(rec), encoding="utf-8")
    log(f"doc: {DOC}")


def stage_file(date: str, log=print) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    key = f"run_record_umlp_d0_{date}"
    if key in text:
        raise SystemExit(f"{key} already filed")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    r = rec["readings"]
    block = {key: {"utc": utc(), "record": RECORD.relative_to(ROOT).as_posix(), "record_sha256": sha256_file(RECORD),
                   "document": DOC.relative_to(ROOT).as_posix(), "masks_sha256": rec["masks"]["sha256"], "masks_mismatches": rec["masks"]["mismatches"],
                   "held_half_read": False, "queries": rec["whole_population"]["queries"],
                   "readings": {k: v["verdict"] for k, v in r.items()},
                   "status_moves": "DECLARED_NOT_RUN -> RUN", "next": "STOP_FOR_REVIEW; Universal-MLP-v2.1 is its own dated declaration"}}
    new = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    if new == text:
        raise SystemExit("status line not found")
    new = new.rstrip(LF) + LF + LF + yaml.safe_dump(block, sort_keys=False, width=200)
    CONFIG.write_text(new, encoding="utf-8")
    log(f"filed {key}; status RUN")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["masks", "read", "doc", "file"], required=True)
    ap.add_argument("--date")
    args = ap.parse_args()
    decl = load_declaration()

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
    if args.stage == "masks":
        verify_inputs(decl)
        stage_masks(decl, log)
    elif args.stage == "read":
        stage_read(decl, log)
    elif args.stage == "doc":
        stage_doc(log)
    else:
        if not args.date:
            raise SystemExit("--stage file needs --date")
        stage_file(args.date, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
