"""UMLP-v2.2A: a zero-initialised, bounded, linear score offset over the frozen u_mlp_v2_mix (configs/umlp_v22a_linear_offset.yaml).

  s_hat = zscore_q(s0) + tau * tanh(w . zscore_q(x)),  w in R^129, w0 = 0, base frozen and in eval mode.

Stages: fit --seed k | eval --dataset D --seeds ... | gate --which A|B | file --date D.
Nothing under configs/universal_v2.yaml, the v2 fits or the v2 eval records is written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("umlp_v21", ROOT / "scripts" / "umlp_v21_coverage.py")
V21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V21)

CONFIG = ROOT / "configs" / "umlp_v22a_linear_offset.yaml"
FROZEN_CONFIGS = V21.FROZEN_CONFIGS + [V21.CONFIG]
OUT = ROOT / "outputs" / "universal_v22a"
FITS = OUT / "fits"
EVAL = OUT / "eval"
V2_FITS, V2_EVAL = V21.V2_FITS, V21.V2_EVAL
BASE = "u_mlp_v2_mix"
NEW = "u_mlp_v22a_off"
CONTROL = BASE
TAU, LAMBDA_P, LAMBDA_D, TOP_N = 1.0, 1.0, 0.01, 5
TRIO, METRICS = V21.TRIO, V21.METRICS
HIT1_TOL = SIDE_TOL = -0.005
LF = chr(10)
key_of, sha256_file, utc = V21.key_of, V21.sha256_file, V21.utc


# ── the model ────────────────────────────────────────────────────────────────


def zscore_q(x, node_query, n):
    from mp_retrieval.m3b_models import segment_zscore
    return segment_zscore(x, node_query, n)


def make_offset_model(base, n_scalars: int):
    import torch
    from torch import nn

    class FrozenBaseOffset(nn.Module):
        arm = NEW

        def __init__(self):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.w = nn.Parameter(torch.zeros(n_scalars))
            self.last = None
            self.base.eval()

        def train(self, mode: bool = True):
            super().train(mode)
            self.base.eval()          # the base never trains and never drops out
            return self

        def forward(self, batch):
            B = batch.n_queries
            with torch.no_grad():
                s0 = self.base(batch)
                s0_hat = zscore_q(s0.unsqueeze(1), batch.node_query, B).squeeze(1)
                z = zscore_q(batch.x, batch.node_query, B)
            delta = TAU * torch.tanh(z @ self.w)
            self.last = (s0_hat, delta)
            return s0_hat + delta

    return FrozenBaseOffset()


def offset_loss(scores, batch, s0_hat, delta):
    import torch
    import torch.nn.functional as Fn
    ptr = batch.qptr.tolist()
    gold = batch.gold.bool()
    sec, prot = [], []
    for q in range(batch.n_queries):
        a, b = int(ptr[q]), int(ptr[q + 1])
        g = gold[a:b]
        gi, ni = torch.nonzero(g).flatten(), torch.nonzero(~g).flatten()
        if gi.numel() == 0 or ni.numel() == 0:
            continue
        s0q, sq = s0_hat[a:b], scores[a:b]
        g1 = gi[torch.argmax(s0q[gi])]
        if gi.numel() >= 2:
            extra = gi[gi != g1]
            disp = ni[torch.topk(s0q[ni], min(TOP_N, int(ni.numel()))).indices]
            sec.append(Fn.softplus(sq[disp].unsqueeze(0) - sq[extra].unsqueeze(1)).mean())
        m0 = s0q[g1] - s0q[ni].max()
        if float(m0) > 0:             # the base's top-1 is a gold: its margin may not shrink
            prot.append(torch.relu(m0 - (sq[g1] - sq[ni].max())))
    zero = scores.new_zeros(())
    l_sec = torch.stack(sec).mean() if sec else zero
    l_prot = torch.stack(prot).mean() if prot else zero
    return l_sec + LAMBDA_P * l_prot + LAMBDA_D * (delta ** 2).mean()


class patched_loss:
    """m3b_train.listwise_loss -> the v2.2A objective (reading the wrapper's cached s0_hat and delta) inside the block only."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        from mp_retrieval import m3b_train
        self.mod, self.orig = m3b_train, m3b_train.listwise_loss

        def objective(scores, batch):
            s0_hat, delta = self.model.last
            return offset_loss(scores, batch, s0_hat, delta)
        objective.v22a = True
        m3b_train.listwise_loss = objective
        return objective

    def __exit__(self, *exc):
        self.mod.listwise_loss = self.orig
        return False


def state_digest(module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_base(U, torch, inputs, bank, seed: int):
    k = key_of(BASE, seed)
    rec = json.loads((V2_FITS / f"{k}.json").read_text(encoding="utf-8"))
    if sha256_file(V2_FITS / f"{k}.pt") != rec["state_sha256"]:
        raise SystemExit(f"{k}.pt: not the recorded sha256; hard stop")
    m = U.make_model(BASE, inputs, bank)
    m.load_state_dict(torch.load(V2_FITS / f"{k}.pt", map_location="cpu"))
    m.eval()
    return m, rec


# ── stage: fit ───────────────────────────────────────────────────────────────


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
    torch, U = V21.load_runner(threads)
    before = {p: sha256_file(p) for p in FROZEN_CONFIGS}
    cfg, cfg_m3b, _ = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    training = U.training_rule_v2(cfg, cfg_m3b)
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    contexts, _, _, bank = U.open_contexts_v2(cfg_m3b, list(TRIO), m3b_compile)
    carves = U.open_carves_v2(contexts, inputs)
    base, base_rec = load_base(U, torch, inputs, bank, seed)
    if base_rec["training"] != training:
        raise SystemExit(f"training rule differs from {key_of(BASE, seed)}'s record; hard stop")
    base_digest = state_digest(base)
    torch.manual_seed(seed)
    model = make_offset_model(base, inputs["n_scalars"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != inputs["n_scalars"]:
        raise SystemExit(f"{trainable} trainable parameters, declared {inputs['n_scalars']}; hard stop")
    log(f"== fit {key}: base {key_of(BASE, seed)} frozen ({base_digest[:12]}), {trainable} trainable; tau {TAU}, lambda_p {LAMBDA_P}, "
        f"lambda_delta {LAMBDA_D}, top {TOP_N}")
    data = U.carves_for_arm(BASE, carves)
    t0 = time.time()
    with patched_loss(model):
        from mp_retrieval import m3b_train
        model, record = m3b_train.fit_model(model, data["fit"], data["select"], seed=seed, arm=NEW,
                                            config={"H": U.HIDDEN, "offset": "v22a_linear", "base": key_of(BASE, seed)},
                                            max_epochs=training["max_epochs"], batches_per_epoch=training["batches_per_epoch"],
                                            batch_size=training["batch_size"], patience=training["patience"], lr=training["lr"],
                                            weight_decay=training["weight_decay"], clip=training["clip"], dataset_draw=training["dataset_draw"],
                                            epoch_limit_s=training["epoch_limit_s"], pack_workers=U.PACK["workers"], prefetch_depth=U.PACK["depth"],
                                            checkpoint=FITS / f"{key}.ckpt", log=log)
    from mp_retrieval import m3b_train
    if getattr(m3b_train.listwise_loss, "v22a", False):
        raise SystemExit("the listwise loss was not restored; hard stop")
    if state_digest(model.base) != base_digest:
        raise SystemExit("the base weights changed during the fit; hard stop")
    torch.save(model.state_dict(), FITS / f"{key}.pt")
    w = model.w.detach().cpu().numpy().astype(np.float64)
    names = [n for n in sorted(U.IDX, key=U.IDX.get)]
    col_names = [names[i] for i in inputs["column_indices"]]
    out = {**asdict(record), "key": key, "base": key_of(BASE, seed), "base_state_sha256": base_rec["state_sha256"], "base_digest": base_digest,
           "trainable_parameters": trainable, "tau": TAU, "lambda_p": LAMBDA_P, "lambda_delta": LAMBDA_D, "top_n": TOP_N, "training": training,
           "w": dict(zip(col_names, [round(float(x), 6) for x in w])), "w_l2": round(float(np.linalg.norm(w)), 6),
           "utc": utc(), "threads": torch.get_num_threads(), "pack_workers": U.PACK["workers"], "prefetch_depth": U.PACK["depth"],
           "peak_rss_bytes": U.M3B_RUN.peak_rss_bytes(), "state_sha256": sha256_file(FITS / f"{key}.pt"), "wall_seconds_this_process": round(time.time() - t0, 1)}
    for p, h in before.items():
        if sha256_file(p) != h:
            raise SystemExit(f"{p.name} changed during the fit; hard stop")
    rec_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (FITS / f"{key}.ckpt").unlink(missing_ok=True)
    log(f"   {key}: best epoch {record.best_epoch} select macro R@5 {record.best_select_macro_recall5:.4f} in {record.seconds:.0f}s; |w| {out['w_l2']}")
    return out


# ── stage: eval ──────────────────────────────────────────────────────────────


def stage_eval(name: str, seeds: list[int], threads: int, log=print) -> None:
    path = EVAL / f"{name}.npz"
    if path.exists():
        log(f"{path} exists; not rescored")
        return
    torch, U = V21.load_runner(threads)
    before = {p: sha256_file(p) for p in FROZEN_CONFIGS}
    cfg, cfg_m3b, cfg_h = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    contexts, handles, pkg, bank = U.open_contexts_v2(cfg_m3b, [name], m3b_compile)
    context, ds = contexts[name], handles[name]
    models, controls, pair_of = {}, [], {}
    for seed in seeds:
        kc, kn = key_of(CONTROL, seed), key_of(NEW, seed)
        base, _ = load_base(U, torch, inputs, bank, seed)
        models[kc] = base
        controls.append(kc)
        rec = json.loads((FITS / f"{kn}.json").read_text(encoding="utf-8"))
        if sha256_file(FITS / f"{kn}.pt") != rec["state_sha256"]:
            raise SystemExit(f"{kn}.pt: not the recorded sha256; hard stop")
        base2, _ = load_base(U, torch, inputs, bank, seed)
        off = make_offset_model(base2, inputs["n_scalars"])
        off.load_state_dict(torch.load(FITS / f"{kn}.pt", map_location="cpu"))
        if state_digest(off.base) != rec["base_digest"]:
            raise SystemExit(f"{kn}: base inside the offset checkpoint is not the frozen base; hard stop")
        off.eval()
        models[kn] = off
        pair_of[kn] = kc
    half_stored, stored = V21.stored_control(name, controls)
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
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, U.M3B_RUN.load_script("m3b_contract"))[0]
    n = len(gi)
    sizes = np.asarray([p.size for p in prep.pools])
    chunk = max(1, int(24000 // max(sizes.mean(), 1)))
    columns = inputs["column_indices"]
    res = {f"{k}/{m}": np.zeros(n) for k in models for m in METRICS}
    for kn in pair_of:
        for d in ("mean_abs_delta", "top1_changed", "top5_set_changed"):
            res[f"{kn}/{d}"] = np.zeros(n)
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
            scores = {}
            for k, model in models.items():
                s = model(U.arm_view(model, batch, inputs))
                scores[k] = s.cpu().numpy().astype(np.float64)
                delta = model.last[1].cpu().numpy() if k in pair_of else None
                for jj, j in enumerate(idx):
                    a, b = ptr[jj], ptr[jj + 1]
                    r = U.rank_metrics(scores[k][a:b], gold_locals[jj], int(pop.golds[j].size))
                    for m in METRICS:
                        res[f"{k}/{m}"][j] = r[m]
                        if k in controls and r[m] != stored[f"{k}/{m}"][j]:
                            raise SystemExit(f"query {pop.ids[j]} (row {gi[j]}), {k} {m}: forward pass {r[m]} != stored {stored[f'{k}/{m}'][j]}; hard stop")
                    if delta is not None:
                        res[f"{k}/mean_abs_delta"][j] = float(np.abs(delta[a:b]).mean())
            for kn, kc in pair_of.items():
                for jj, j in enumerate(idx):
                    a, b = ptr[jj], ptr[jj + 1]
                    o_new = np.argsort(-scores[kn][a:b], kind="stable")
                    o_old = np.argsort(-scores[kc][a:b], kind="stable")
                    res[f"{kn}/top1_changed"][j] = float(o_new[0] != o_old[0])
                    res[f"{kn}/top5_set_changed"][j] = float(set(o_new[:5].tolist()) != set(o_old[:5].tolist()))
            if (start // chunk) % 10 == 0:
                log(f"   {name}: {min(start + chunk, n)}/{n} scored, {time.time() - t0:.0f}s, controls equal to stored so far")
    for p, h in before.items():
        if sha256_file(p) != h:
            raise SystemExit(f"{p.name} changed during the run; hard stop")
    EVAL.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, row=gi.astype(np.int64), **res)
    tmp.replace(path)
    meta = {"utc": utc(), "dataset": name, "queries": n, "models": list(models), "controls_integrity": controls, "mismatches": 0,
            "seconds": round(time.time() - t0, 1), "chunk_queries": chunk, "threads": threads, "sha256": sha256_file(path)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"eval {name}: {n} V2_GATE queries, {len(models)} models, 0 control mismatches, {meta['seconds']}s")


# ── stage: gate / file ───────────────────────────────────────────────────────


def load_eval(name: str) -> dict:
    with np.load(EVAL / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def mean_over(rows: dict, arm: str, seeds: list[int], m: str) -> np.ndarray:
    return np.mean([rows[f"{key_of(arm, s)}/{m}"] for s in seeds], axis=0)


def paired(rows: dict, seeds: list[int], U) -> dict:
    return {m: U.paired_bootstrap(mean_over(rows, NEW, seeds, m), mean_over(rows, CONTROL, seeds, m)) for m in METRICS
            if m not in ("gold_in_pool", "gold_total", "pool_size")}


def levels(rows: dict, seeds: list[int]) -> dict:
    out = {arm: {f"s{s}": {m: round(float(rows[f"{key_of(arm, s)}/{m}"].mean()), 4) for m in METRICS
                           if m not in ("gold_in_pool", "gold_total", "pool_size", "first_gold_rank")} for s in seeds} for arm in (NEW, CONTROL)}
    for s in seeds:
        out[NEW][f"s{s}"].update({d: round(float(rows[f"{key_of(NEW, s)}/{d}"].mean()), 4)
                                  for d in ("mean_abs_delta", "top1_changed", "top5_set_changed")})
    return out


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
               "paired_v22a_minus_control": p, "levels": levels(rows, [0]), "conditions": cond, "pass": all(cond.values()),
               "reading": "V22A_STAGE_A_PASS" if all(cond.values()) else "V22A_STAGE_A_FAIL"}
    else:
        per, cond = {}, {}
        for name in TRIO:
            rows = load_eval(name)
            p = paired(rows, [0, 1, 2], U)
            per[name] = {"queries": int(rows["row"].size), "paired_seed_mean_v22a_minus_control": p, "levels": levels(rows, [0, 1, 2])}
            cond[name] = (twowiki_conditions(p) if name == "2wiki" else
                          {"recall@5_mean_ge_-0.005": p["recall@5"]["mean"] >= SIDE_TOL, "hit@1_mean_ge_-0.005": p["hit@1"]["mean"] >= SIDE_TOL})
        wiki_ok = all(cond["2wiki"].values())
        side_ok = all(all(cond[n].values()) for n in ("metaqa", "squad"))
        reading = "V22A_CONFIRMED" if wiki_ok and side_ok else "V22A_2WIKI_ONLY" if wiki_ok else "V22A_NOT_CONFIRMED"
        out = {"stage": "B", "utc": utc(), "seeds": [0, 1, 2], "per_dataset": per, "conditions": cond, "reading": reading}
    (OUT / f"gate_{which}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(json.dumps({k: out[k] for k in ("stage", "conditions", "reading")}, indent=1))
    return out


def stage_file(date: str, log=print) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_umlp_v22a_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    gates = {s: json.loads((OUT / f"gate_{s}.json").read_text(encoding="utf-8")) for s in ("A", "B") if (OUT / f"gate_{s}.json").exists()}
    fits = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(FITS.glob("*.json"))}
    rec = {"utc": utc(), "held_half_read": False,
           "fits": {k: {"best_epoch": f["best_epoch"], "epochs_run": f["epochs_run"], "select_macro_recall@5": round(f["best_select_macro_recall5"], 4),
                        "seconds": f["seconds"], "w_l2": f["w_l2"], "state_sha256": f["state_sha256"]} for k, f in fits.items()},
           "gates": {s: {"reading": g["reading"], "conditions": g["conditions"], "sha256": sha256_file(OUT / f"gate_{s}.json")} for s, g in gates.items()},
           "document": "docs/UMLP_V22A_LINEAR_OFFSET.md", "status_moves": "DECLARED_NOT_RUN -> RUN"}
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
        stage_eval(a.dataset, a.seeds, a.threads, log=log)
    elif a.stage == "gate":
        stage_gate(a.which, log=log)
    else:
        stage_file(a.date, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
