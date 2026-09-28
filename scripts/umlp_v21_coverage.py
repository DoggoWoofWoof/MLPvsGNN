"""UMLP-v2.1: the frozen u_mlp_v2_mix trained with L = listwise + 1.0 * coverage (configs/umlp_v21_coverage_objective.yaml).

Stages
  fit   --seed S          one trio fit of u_mlp_v21_cov through the pinned fit_model, the loss swapped for this process only
  eval  --dataset D       V2_GATE rows only: every v2.1 fit present and the frozen control seeds re-scored in one pass;
                          every control forward pass must reproduce its stored per-query metrics (hard stop otherwise)
  gate                    stage A (2wiki, seed 0) or stage B (trio, seed means), the declared conditions only
  file --date D           run record appended to the declaration, status DECLARED_NOT_RUN -> RUN

Nothing under configs/universal_v2.yaml, the v2 fits or the v2 eval records is written.
"""

from __future__ import annotations

import argparse
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
CONFIG = ROOT / "configs" / "umlp_v21_coverage_objective.yaml"
FROZEN_CONFIGS = [ROOT / "configs" / n for n in ("universal_v2.yaml", "umlp_d0_2wiki_diagnostics.yaml",
                                                   "umlp_d01_2wiki_feature_readout.yaml", "umlp_d02_joint_separability.yaml")]
OUT = ROOT / "outputs" / "universal_v21"
FITS = OUT / "fits"
EVAL = OUT / "eval"
V2_FITS = ROOT / "outputs" / "universal_v2" / "fits"
V2_EVAL = ROOT / "outputs" / "universal_v2" / "eval"
ARM = "u_mlp_v2_mix"            # the architecture built; the loss is the only change
NEW = "u_mlp_v21_cov"
CONTROL = "u_mlp_v2_mix"
LAMBDA = 1.0
TOP_H = 5
TRIO = ("metaqa", "2wiki", "squad")
METRICS = ("recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "mrr", "ndcg@5", "ndcg@20", "full_coverage@5",
           "full_coverage@20", "first_gold_rank", "gold_in_pool", "gold_total", "pool_size")
GATE_METRICS = ("recall@5", "hit@1", "full_coverage@5")
HIT1_TOL = -0.005
SIDE_TOL = -0.005
LF = chr(10)


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key_of(arm: str, seed: int) -> str:
    return f"{arm}__H128__s{seed}"


# ── the objective ────────────────────────────────────────────────────────────


def coverage_loss(scores, batch, top_h: int = TOP_H):
    """Per query with >= 2 in-pool golds: mean over golds g, mean over the top_h highest-scoring in-pool non-golds n
    (selected on detached scores) of softplus(s_n - s_g); other queries contribute 0. Summed and divided by the number
    of queries with an in-pool gold (the listwise loss's denominator)."""
    import torch
    import torch.nn.functional as Fn
    ptr = batch.qptr.tolist() if hasattr(batch.qptr, "tolist") else list(batch.qptr)
    gold = batch.gold.bool()
    total = scores.new_zeros(())
    has = 0
    for q in range(batch.n_queries):
        a, b = int(ptr[q]), int(ptr[q + 1])
        g = gold[a:b]
        n_g = int(g.sum())
        if n_g == 0:
            continue
        has += 1
        if n_g < 2 or n_g == b - a:
            continue
        s = scores[a:b]
        s_g, s_n = s[g], s[~g]
        k = min(top_h, int(s_n.numel()))
        top = torch.topk(s_n.detach(), k).indices
        total = total + Fn.softplus(s_n[top].unsqueeze(0) - s_g.unsqueeze(1)).mean()
    if has == 0:
        raise RuntimeError("training batch has no in-pool gold")
    return total / has


def make_objective(listwise):
    def objective(scores, batch):
        return listwise(scores, batch) + LAMBDA * coverage_loss(scores, batch)
    objective.v21 = True
    return objective


class patched_loss:
    """Swap m3b_train.listwise_loss (the name fit_model calls) for the v2.1 objective inside the block only."""

    def __enter__(self):
        from mp_retrieval import m3b_train
        self.mod, self.orig = m3b_train, m3b_train.listwise_loss
        m3b_train.listwise_loss = make_objective(self.orig)
        return m3b_train.listwise_loss

    def __exit__(self, *exc):
        self.mod.listwise_loss = self.orig
        return False


# ── stage: fit ───────────────────────────────────────────────────────────────


def load_runner(threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import universal_v2_run as U   # frozen runner, imported not edited
    torch.set_num_threads(threads)
    return torch, U


def stage_fit(seed: int, threads: int, log=print) -> dict:
    key = key_of(NEW, seed)
    FITS.mkdir(parents=True, exist_ok=True)
    rec_path = FITS / f"{key}.json"
    if rec_path.exists():
        log(f"{key}: record exists, not repeated")
        return json.loads(rec_path.read_text(encoding="utf-8"))
    if seed != 0:
        gate = OUT / "gate_A.json"
        if not gate.exists() or not json.loads(gate.read_text(encoding="utf-8"))["pass"]:
            raise SystemExit(f"seed {seed}: stage B runs only after a stage-A pass (gate_A.json)")
    torch, U = load_runner(threads)
    before = {p: sha256_file(p) for p in FROZEN_CONFIGS}
    cfg, cfg_m3b, _ = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    training = U.training_rule_v2(cfg, cfg_m3b)
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    contexts, _, _, bank = U.open_contexts_v2(cfg_m3b, list(TRIO), m3b_compile)
    carves = U.open_carves_v2(contexts, inputs)
    control = json.loads((V2_FITS / f"{key_of(CONTROL, 0)}.json").read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    model = U.make_model(ARM, inputs, bank)
    params = U.parameter_count(model)
    frozen = {"parameters": params, "hidden": U.HIDDEN, "contract_block": inputs["contract_block"], "columns": inputs["n_scalars"],
              "core_sha256": inputs["core_sha256"], "base": inputs["base"], "training": training,
              "relation_bank": {"rows": bank.n_rows, "sha256": bank.sha256, "k_rel": U.K_REL}}
    differs = {k: {"v21": v, "control_s0": control.get(k)} for k, v in frozen.items() if control.get(k) != v}
    if differs:
        raise SystemExit(f"{key}: protocol differs from the control's seed-0 record: {differs}; hard stop")
    log(f"== fit {key}: {params} parameters; protocol equal to {key_of(CONTROL, 0)} on {sorted(frozen)}; loss listwise + {LAMBDA} * coverage(top {TOP_H})")
    data = U.carves_for_arm(ARM, carves)
    t0 = time.time()
    with patched_loss() as obj:
        from mp_retrieval import m3b_train
        assert m3b_train.listwise_loss is obj and getattr(obj, "v21", False)
        model, record = m3b_train.fit_model(model, data["fit"], data["select"], seed=seed, arm=NEW, config={"H": U.HIDDEN, "loss": "v21_coverage"},
                                            max_epochs=training["max_epochs"], batches_per_epoch=training["batches_per_epoch"],
                                            batch_size=training["batch_size"], patience=training["patience"], lr=training["lr"],
                                            weight_decay=training["weight_decay"], clip=training["clip"], dataset_draw=training["dataset_draw"],
                                            epoch_limit_s=training["epoch_limit_s"], pack_workers=U.PACK["workers"], prefetch_depth=U.PACK["depth"],
                                            checkpoint=FITS / f"{key}.ckpt", log=log)
    from mp_retrieval import m3b_train
    if getattr(m3b_train.listwise_loss, "v21", False):
        raise SystemExit("the listwise loss was not restored; hard stop")
    torch.save(model.state_dict(), FITS / f"{key}.pt")
    from dataclasses import asdict
    out = {**asdict(record), "key": key, "architecture": ARM, **frozen,
           "loss": {"form": "listwise + lambda * coverage", "lambda": LAMBDA, "top_h": TOP_H, "declaration": str(CONFIG.relative_to(ROOT))},
           "utc": utc(), "threads": torch.get_num_threads(), "pack_workers": U.PACK["workers"], "prefetch_depth": U.PACK["depth"],
           "peak_rss_bytes": U.M3B_RUN.peak_rss_bytes(), "state_sha256": sha256_file(FITS / f"{key}.pt"), "wall_seconds": round(time.time() - t0, 1)}
    for p, h in before.items():
        if sha256_file(p) != h:
            raise SystemExit(f"{p.name} changed during the fit; hard stop")
    rec_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (FITS / f"{key}.ckpt").unlink(missing_ok=True)
    log(f"   {key}: best epoch {record.best_epoch} select macro R@5 {record.best_select_macro_recall5:.4f} in {record.seconds:.0f}s")
    return out


# ── stage: eval ──────────────────────────────────────────────────────────────


def stored_control(name: str, keys: list[str]) -> tuple[np.ndarray, dict]:
    """The frozen control's stored per-query metrics on V2_GATE rows (held rows dropped as read)."""
    with np.load(V2_EVAL / f"{name}.npz") as z:
        half = z["half"].astype(bool)
    out = {}
    for f in (V2_EVAL / f"{name}.npz", V2_EVAL / f"{name}__more_69caa3fe.npz"):
        with np.load(f) as z:
            for k in keys:
                for m in METRICS:
                    if f"{k}/{m}" in z.files:
                        out[f"{k}/{m}"] = z[f"{k}/{m}"][half]
    missing = [f"{k}/{m}" for k in keys for m in METRICS if f"{k}/{m}" not in out]
    if missing:
        raise SystemExit(f"stored control metrics missing: {missing[:4]}; refusing")
    return half, out


def stage_eval(name: str, seeds: list[int], threads: int, shard: tuple[int, int] | None, log=print) -> None:
    tag = name if shard is None else f"{name}__shard{shard[0]}of{shard[1]}"
    path = EVAL / f"{tag}.npz"
    if path.exists():
        log(f"{path} exists; not rescored")
        return
    torch, U = load_runner(threads)
    before = {p: sha256_file(p) for p in FROZEN_CONFIGS}
    cfg, cfg_m3b, cfg_h = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    contexts, handles, pkg, bank = U.open_contexts_v2(cfg_m3b, [name], m3b_compile)
    context, ds = contexts[name], handles[name]
    models, controls = {}, []
    for seed in seeds:
        for arm, fits in ((NEW, FITS), (CONTROL, V2_FITS)):
            k = key_of(arm, seed)
            rec = json.loads((fits / f"{k}.json").read_text(encoding="utf-8"))
            if sha256_file(fits / f"{k}.pt") != rec["state_sha256"]:
                raise SystemExit(f"{k}.pt: not the recorded sha256; hard stop")
            m = U.make_model(ARM, inputs, bank)
            m.load_state_dict(torch.load(fits / f"{k}.pt", map_location="cpu"))
            m.eval()
            models[k] = m
            if arm == CONTROL:
                controls.append(k)
    half_stored, stored = stored_control(name, controls)
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
    sel = np.arange(gi.size) if shard is None else np.arange(gi.size)[np.arange(gi.size) % shard[1] == shard[0]]
    gi = gi[sel]
    stored = {k: v[sel] for k, v in stored.items()}
    pop.ids, pop.idx, pop.golds = [pop.ids[i] for i in gi], pop.idx[gi], [pop.golds[i] for i in gi]
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract := U.M3B_RUN.load_script("m3b_contract"))[0]
    n = len(gi)
    sizes = np.asarray([p.size for p in prep.pools])
    chunk = max(1, int(24000 // max(sizes.mean(), 1)))
    columns = inputs["column_indices"]
    res = {f"{k}/{m}": np.zeros(n) for k in models for m in METRICS}
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
            for k, model in models.items():
                s = model(U.arm_view(model, batch, inputs)).cpu().numpy().astype(np.float64)
                for jj, j in enumerate(idx):
                    r = U.rank_metrics(s[ptr[jj]:ptr[jj + 1]], gold_locals[jj], int(pop.golds[j].size))
                    for m in METRICS:
                        res[f"{k}/{m}"][j] = r[m]
                        if k in controls and r[m] != stored[f"{k}/{m}"][j]:
                            raise SystemExit(f"query {pop.ids[j]} (row {gi[j]}), {k} {m}: forward pass {r[m]} != stored {stored[f'{k}/{m}'][j]}; hard stop")
            if (start // chunk) % 10 == 0:
                log(f"   {name}: {min(start + chunk, n)}/{n} scored, {time.time() - t0:.0f}s, controls equal to stored so far")
    for p, h in before.items():
        if sha256_file(p) != h:
            raise SystemExit(f"{p.name} changed during the run; hard stop")
    EVAL.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, row=gi.astype(np.int64), **res)
    tmp.replace(path)
    meta = {"utc": utc(), "dataset": name, "shard": shard, "queries": n, "models": list(models), "controls_integrity": controls,
            "mismatches": 0, "seconds": round(time.time() - t0, 1), "chunk_queries": chunk, "threads": threads, "sha256": sha256_file(path)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"eval {tag}: {n} V2_GATE queries, {len(models)} models, 0 control mismatches, {meta['seconds']}s")


def load_eval(name: str) -> dict:
    """One dataset's V2_GATE arrays: the unsharded record, or the shards concatenated in row order."""
    if (EVAL / f"{name}.npz").exists():
        parts = [EVAL / f"{name}.npz"]
    else:
        parts = sorted(EVAL.glob(f"{name}__shard*of*.npz"))
        if not parts:
            raise SystemExit(f"{name}: no eval record")
        n_sh = int(parts[0].stem.split("of")[-1])
        if len(parts) != n_sh:
            raise SystemExit(f"{name}: {len(parts)} of {n_sh} shards")
    zs = [dict(np.load(p)) for p in parts]
    keys = set(zs[0])
    if any(set(z) != keys for z in zs):
        raise SystemExit(f"{name}: shards carry different models")
    cat = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    order = np.argsort(cat["row"], kind="stable")
    return {k: v[order] for k, v in cat.items()}


# ── stage: gate ──────────────────────────────────────────────────────────────


def mean_over(rows: dict, arm: str, seeds: list[int], m: str) -> np.ndarray:
    return np.mean([rows[f"{key_of(arm, s)}/{m}"] for s in seeds], axis=0)


def paired(rows: dict, seeds: list[int], U) -> dict:
    return {m: U.paired_bootstrap(mean_over(rows, NEW, seeds, m), mean_over(rows, CONTROL, seeds, m)) for m in METRICS
            if m not in ("gold_in_pool", "gold_total", "pool_size")}


def levels(rows: dict, seeds: list[int]) -> dict:
    return {arm: {f"s{s}": {m: round(float(rows[f"{key_of(arm, s)}/{m}"].mean()), 4) for m in METRICS
                            if m not in ("gold_in_pool", "gold_total", "pool_size", "first_gold_rank")} for s in seeds} for arm in (NEW, CONTROL)}


def twowiki_conditions(p: dict) -> dict:
    return {"full_coverage@5_lower_gt_0": p["full_coverage@5"]["low"] > 0, "recall@5_mean_ge_0": p["recall@5"]["mean"] >= 0,
            "hit@1_mean_ge_-0.005": p["hit@1"]["mean"] >= HIT1_TOL}


def stage_gate(which: str, log=print) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import universal_v2_run as U
    if which == "A":
        rows = load_eval("2wiki")
        p = paired(rows, [0], U)
        cond = twowiki_conditions(p)
        out = {"stage": "A", "utc": utc(), "population": "2wiki V2_GATE", "queries": int(rows["row"].size), "seeds": [0],
               "paired_v21_minus_control": p, "levels": levels(rows, [0]), "conditions": cond, "pass": all(cond.values()),
               "reading": "V21_STAGE_A_PASS" if all(cond.values()) else "V21_STAGE_A_FAIL"}
        (OUT / "gate_A.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    else:
        per, cond = {}, {}
        for name in TRIO:
            rows = load_eval(name)
            p = paired(rows, [0, 1, 2], U)
            per[name] = {"queries": int(rows["row"].size), "paired_seed_mean_v21_minus_control": p, "levels": levels(rows, [0, 1, 2])}
            cond[name] = (twowiki_conditions(p) if name == "2wiki" else
                          {"recall@5_mean_ge_-0.005": p["recall@5"]["mean"] >= SIDE_TOL, "hit@1_mean_ge_-0.005": p["hit@1"]["mean"] >= SIDE_TOL})
        wiki_ok = all(cond["2wiki"].values())
        side_ok = all(all(cond[n].values()) for n in ("metaqa", "squad"))
        reading = "V21_CONFIRMED" if wiki_ok and side_ok else "V21_2WIKI_ONLY" if wiki_ok else "V21_NOT_CONFIRMED"
        out = {"stage": "B", "utc": utc(), "seeds": [0, 1, 2], "per_dataset": per, "conditions": cond, "reading": reading}
        (OUT / "gate_B.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(json.dumps({k: out[k] for k in ("stage", "conditions", "reading")}, indent=1))
    return out


# ── stage: file ──────────────────────────────────────────────────────────────


def stage_file(date: str, log=print) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_umlp_v21_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    gates = {s: json.loads((OUT / f"gate_{s}.json").read_text(encoding="utf-8")) for s in ("A", "B") if (OUT / f"gate_{s}.json").exists()}
    fits = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(FITS.glob("*.json"))}
    rec = {"utc": utc(), "held_half_read": False,
           "fits": {k: {"best_epoch": f["best_epoch"], "epochs_run": f["epochs_run"], "select_macro_recall@5": round(f["best_select_macro_recall5"], 4),
                        "seconds": f["seconds"], "state_sha256": f["state_sha256"]} for k, f in fits.items()},
           "gates": {s: {"reading": g["reading"], "conditions": g["conditions"], "sha256": sha256_file(OUT / f"gate_{s}.json")} for s, g in gates.items()},
           "document": "docs/UMLP_V21_COVERAGE_OBJECTIVE.md", "status_moves": "DECLARED_NOT_RUN -> RUN"}
    block = yaml.safe_dump({key: rec}, sort_keys=False, width=160)
    text = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + block, encoding="utf-8")
    log(f"filed {key}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "eval", "gate", "file"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--shard", default=None, help="i/n: every n-th V2_GATE query from i")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--which", default="A")
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"_run_{a.stage}.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + LF)

    if a.stage == "fit":
        stage_fit(a.seed, a.threads, log=log)
    elif a.stage == "eval":
        shard = None if a.shard is None else tuple(int(x) for x in a.shard.split("/"))
        stage_eval(a.dataset, a.seeds, a.threads, shard, log=log)
    elif a.stage == "gate":
        stage_gate(a.which, log=log)
    else:
        stage_file(a.date, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
