"""UMLP-D0.2: can the frozen compiled representation jointly separate a GNN-recovered extra gold from the twin's
top-5 displacers? (configs/umlp_d02_joint_separability.yaml)

    python scripts/umlp_d02_joint_separability.py --stage run    # cross-fitted probes A/B/C per twin seed -> record.json
    python scripts/umlp_d02_joint_separability.py --stage doc
    python scripts/umlp_d02_joint_separability.py --stage file --date 2026_09_29

Reads only the D0.1 sidecar (V2_GATE rows). Probes are diagnostic; no retrieval model is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "umlp_d02_joint_separability.yaml"
FROZEN_CONFIGS = [ROOT / "configs" / n for n in ("universal_v2.yaml", "umlp_d0_2wiki_diagnostics.yaml", "umlp_d01_2wiki_feature_readout.yaml")]
OUT = ROOT / "outputs" / "umlp_d02"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "UMLP_D02_JOINT_SEPARABILITY.md"
LF = chr(10)
FOLDS = 5
BAR = 0.60
HIDDEN, STEPS, LR, WD = 64, 500, 1e-3, 1e-4
TASKS = {"primary": "G_miss_mp", "secondary": "G_miss"}
FORBIDDEN = ("score", "rank", "residual", "gnn", "arm", "dataset")

_spec = importlib.util.spec_from_file_location("umlp_d01", ROOT / "scripts" / "umlp_d01_feature_readout.py")
D01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D01)


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    return D01.sha256_file(path)


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def fold_of(query_id: str) -> int:
    return int(hashlib.sha256(query_id.encode("utf-8")).hexdigest(), 16) % FOLDS


def feature_sets(z: dict, screen: dict) -> dict[str, list[str]]:
    names = [str(s) for s in z["column_names"]]
    surviving, raw = list(screen["surviving"]), list(screen["columns"])
    if len(surviving) != 129 or len(raw) != 164 or set(surviving) != set(map(str, z["read_columns"])):
        raise SystemExit("the retained / raw column sets are not the filed 129 / 164; refusing")
    missing = [c for c in raw if c not in names]
    if missing:
        raise SystemExit(f"raw columns missing from the sidecar: {missing[:4]}; refusing")
    sets = {"A": surviving + ["__base__"], "B": surviving + ["__base__"], "C": raw + ["__base__"]}
    for k, cols in sets.items():
        bad = [c for c in cols if c != "__base__" and any(f in c for f in ("gnn", "twin"))]
        if bad:
            raise SystemExit(f"probe {k}: forbidden inputs {bad}")
    return sets


def build_pairs(qi: np.ndarray, pos: np.ndarray, neg: np.ndarray, top_gold: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gold_row, neg_row, query) for every same-query pair of the task, analysed queries only."""
    G, N, Q = [], [], []
    order = np.argsort(qi, kind="stable")
    qs, starts = np.unique(qi[order], return_index=True)
    ends = np.append(starts[1:], order.size)
    for q, a, b in zip(qs, starts, ends):
        if not top_gold[q]:
            continue
        idx = order[a:b]
        gp, gn = idx[pos[idx]], idx[neg[idx]]
        for g in gp:
            for n in gn:
                G.append(g)
                N.append(n)
                Q.append(q)
    return np.asarray(G, dtype=np.int64), np.asarray(N, dtype=np.int64), np.asarray(Q, dtype=np.int64)


def query_weights(Q: np.ndarray) -> np.ndarray:
    """Per pair weight so every query totals 1 over its 2 * n_pairs oriented rows (returned per pair, per orientation)."""
    _, inv, counts = np.unique(Q, return_inverse=True, return_counts=True)
    return 1.0 / (2.0 * counts[inv])


def fit_linear(Xg, Xn, w):
    from sklearn.linear_model import LogisticRegression
    d = Xg - Xn
    X = np.concatenate([d, -d])
    y = np.concatenate([np.ones(len(d)), np.zeros(len(d))])
    m = LogisticRegression(penalty="l2", C=1.0, fit_intercept=False, solver="lbfgs", max_iter=5000)
    m.fit(X, y, sample_weight=np.concatenate([w, w]))
    coef = m.coef_.ravel().copy()
    return (lambda X_: X_ @ coef), int(coef.size), {"n_iter": int(np.max(m.n_iter_)), "converged": bool(np.max(m.n_iter_) < 5000)}


def fit_mlp(Xg, Xn, w):
    import torch
    torch.manual_seed(0)
    k = Xg.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(k, HIDDEN), torch.nn.GELU(), torch.nn.Linear(HIDDEN, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    tg, tn = torch.tensor(Xg, dtype=torch.float32), torch.tensor(Xn, dtype=torch.float32)
    tw = torch.tensor(w, dtype=torch.float32)
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(STEPS):
        opt.zero_grad()
        logit = (net(tg) - net(tn)).squeeze(-1)
        loss = (tw * loss_fn(logit, torch.ones_like(logit), reduction="none")).sum() + (tw * loss_fn(-logit, torch.zeros_like(logit), reduction="none")).sum()
        loss.backward()
        opt.step()
    net.eval()

    def score(X_):
        with torch.no_grad():
            return net(torch.tensor(X_, dtype=torch.float32)).squeeze(-1).numpy().astype(np.float64)
    return score, int(sum(p.numel() for p in net.parameters())), {"final_loss": round(float(loss.item()), 6)}


def cross_fit(V: np.ndarray, G: np.ndarray, N: np.ndarray, Q: np.ndarray, fold_q: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    """Out-of-fold candidate scores for every paired candidate row; normalisation fitted on training folds only."""
    oof = np.full(V.shape[0], np.nan)
    w_all = query_weights(Q)
    info = {"params": None, "folds": []}
    fold_pair = fold_q[Q]
    for f in range(FOLDS):
        tr, te = fold_pair != f, fold_pair == f
        if not te.any():
            continue
        rows_tr = np.unique(np.concatenate([G[tr], N[tr]]))
        mu, sd = V[rows_tr].mean(axis=0), V[rows_tr].std(axis=0)
        sd[sd < 1e-12] = 1.0
        Z = (V - mu) / sd
        w = w_all[tr]   # as declared: every training query totals weight 1 (no renormalisation)
        fit = fit_linear if kind == "linear" else fit_mlp
        score, params, extra = fit(Z[G[tr]], Z[N[tr]], w)
        rows_te = np.unique(np.concatenate([G[te], N[te]]))
        if np.isin(rows_te, rows_tr).any():
            raise SystemExit("a test-fold candidate is in the training folds; refusing")
        oof[rows_te] = score(Z[rows_te])
        info["params"] = params
        info["folds"].append({"fold": f, "train_pairs": int(tr.sum()), "test_pairs": int(te.sum()), **extra})
    return oof, info


def evaluate(scores: np.ndarray, G: np.ndarray, N: np.ndarray, Q: np.ndarray) -> dict:
    """Query-weighted within-query pair-AUC with the pilot bootstrap, plus unweighted pair accuracy."""
    sg, sn = scores[G], scores[N]
    correct = (sg > sn) + 0.5 * (sg == sn)
    qs, inv = np.unique(Q, return_inverse=True)
    per_q = np.bincount(inv, weights=correct) / np.bincount(inv)
    boot = D01.bootstrap_mean(per_q[:, None])
    return {"pair_auc": round(float(boot["mean"][0]), 4), "low": round(float(boot["low"][0]), 4), "high": round(float(boot["high"][0]), 4),
            "pair_accuracy": round(float(correct.mean()), 4), "queries": int(qs.size), "pairs": int(G.size), "positive_golds": int(np.unique(G).size),
            "clears_bar": bool(boot["low"][0] > BAR)}


def run(decl: dict, log=print) -> dict:
    for key, item in decl["inputs"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise SystemExit(f"{item['path']}: not the pinned sha256; hard stop")
    before = {p.name: sha256_file(p) for p in FROZEN_CONFIGS}
    with np.load(ROOT / decl["inputs"]["d01_sidecar"]["path"]) as f:
        z = {k: f[k] for k in f.files}
    screen = json.loads((ROOT / decl["inputs"]["feature_screen"]["path"]).read_text(encoding="utf-8"))
    ids = json.loads((ROOT / decl["inputs"]["query_ids"]["path"]).read_text(encoding="utf-8"))
    d01 = yaml.safe_load((ROOT / "configs" / "umlp_d01_2wiki_feature_readout.yaml").read_text(encoding="utf-8"))
    with np.load(ROOT / d01["inputs"]["v2_eval_seed0"]["path"]) as e:
        gate_rows = np.flatnonzero(e["half"].astype(bool))
    if not np.array_equal(z["row"], gate_rows):
        raise SystemExit("sidecar rows are not the V2_GATE rows; refusing")
    fold_q = np.asarray([fold_of(ids[int(r)]) for r in z["row"]])
    sets = feature_sets(z, screen)
    names = [str(s) for s in z["column_names"]]
    col = {n: i for i, n in enumerate(names)}
    twin_keys = d01["arms_scored"]["twin"]
    gnn_keys = d01["arms_scored"]["mp_reference"]
    qi = z["query"]
    gnn_rank = np.mean([z[f"{k}/rank"] for k in gnn_keys], axis=0)
    result = {}
    t0 = time.time()
    for twin in twin_keys:
        g, top_gold = D01.groups_for_seed(z, twin, gnn_keys)
        base = z[f"{twin}/base"]
        seed_out = {}
        for task, pos_name in TASKS.items():
            G, N, Q = build_pairs(qi, g[pos_name], g["N_disp"], top_gold)
            refs = {"fixed_base": evaluate(base, G, N, Q),
                    "twin_residual": evaluate(z[f"{twin}/score"] - base, G, N, Q),
                    "twin_score": evaluate(z[f"{twin}/score"], G, N, Q),
                    "gnn_score": evaluate(-gnn_rank, G, N, Q)}
            probes = {}
            for probe, kind in (("A", "linear"), ("B", "mlp"), ("C", "mlp")):
                V = np.stack([base if c == "__base__" else z["X"][:, col[c]].astype(np.float64) for c in sets[probe]], axis=1)
                oof, info = cross_fit(V, G, N, Q, fold_q, kind)
                if np.isnan(oof[np.concatenate([G, N])]).any():
                    raise SystemExit(f"{twin} {task} {probe}: a paired candidate has no out-of-fold score; refusing")
                probes[probe] = {**evaluate(oof, G, N, Q), "inputs": len(sets[probe]), "parameters": info["params"], "folds": info["folds"]}
                log(f"   {twin.split('__')[-1]} {task} probe {probe}: pair-AUC {probes[probe]['pair_auc']} "
                    f"[{probes[probe]['low']}, {probes[probe]['high']}] over {probes[probe]['queries']} queries, {time.time() - t0:.0f}s")
            fold_counts = np.bincount(fold_q[np.unique(Q)], minlength=FOLDS).tolist()
            seed_out[task] = {"references": refs, "probes": probes, "queries_per_fold": fold_counts}
        result[twin] = seed_out
    after = {p.name: sha256_file(p) for p in FROZEN_CONFIGS}
    if after != before:
        raise SystemExit("a frozen config changed during the run; hard stop")
    clears = {p: all(result[t]["primary"]["probes"][p]["clears_bar"] for t in twin_keys) for p in ("A", "B", "C")}
    if clears["A"]:
        reading = "D02_R1_LINEARLY_PRESENT"
    elif clears["B"]:
        reading = "D02_R2_NONLINEARLY_PRESENT"
    elif clears["C"]:
        reading = "D02_R3_SCREENED_INFORMATION"
    else:
        reading = "D02_R4_NOT_RECOVERED_FROM_COMPILED_BASIS"
    feats = {k: {"n": len(v), "sha256": hashlib.sha256(LF.join(v).encode("utf-8")).hexdigest(), "columns": v} for k, v in sets.items()}
    return {"phase": decl["phase"], "utc": utc(), "held_half_read": False, "gat_loaded": False, "seconds": round(time.time() - t0, 1),
            "inputs": {k: v for k, v in decl["inputs"].items()}, "frozen_config_sha256": before, "feature_sets": feats,
            "cross_fit": {"folds": FOLDS, "rule": "sha256(query_id) mod 5", "queries_per_fold_all_gate": np.bincount(fold_q, minlength=FOLDS).tolist()},
            "hyperparameters": {"A": "LogisticRegression l2 C=1 no intercept lbfgs max_iter 5000",
                                "B_C": f"Linear(k,{HIDDEN})-GELU-Linear({HIDDEN},1), AdamW lr {LR} wd {WD}, {STEPS} full-batch steps, seed 0"},
            "per_seed": result, "clears_bar_all_three_seeds": clears, "reading": reading,
            "incidents": ["ruling's 129 / 164 verified against feature_screen.json before filing (surviving 129 == twin read set, columns 164, dropped 35)"]}


def stage_run(decl, log=print):
    if RECORD.exists():
        raise SystemExit(f"{RECORD} exists: run once")
    rec = run(decl, log)
    OUT.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log(f"record {RECORD} sha256 {sha256_file(RECORD)[:12]}; clears {rec['clears_bar_all_three_seeds']}; reading {rec['reading']}")


def f3(c) -> str:
    return f"{c['pair_auc']:.3f} [{c['low']:.3f}, {c['high']:.3f}]"


def render_doc(rec: dict) -> str:
    seeds = list(rec["per_seed"])
    L = ["# UMLP-D0.2: joint separability of the golds the twin misses (2wiki)", "",
         "Analysis only (configs/umlp_d02_joint_separability.yaml). Question: can the frozen compiled representation jointly distinguish an "
         "additional gold recovered by the GNN from the non-golds occupying the twin's top-5 slots? Population: the D0.1 sidecar "
         "(2wiki V2_GATE, integrity-checked); held half not read; M3B GAT not loaded; no retrieval model touched.", "",
         f"**Reading: {rec['reading']}.** Bar: out-of-fold within-query pair-AUC lower 95% bound > {BAR} on all three twin seeds.", ""]
    if rec["reading"].startswith("D02_R4"):
        L += ["No recoverable signal was demonstrated under the registered linear and small-nonlinear probe classes.", ""]
    L += ["## Primary task: G_miss_mp vs N_disp (out-of-fold)", "",
          "| twin seed | queries | golds | pairs | A retained linear | B retained MLP | C raw-164 MLP |", "|---|---:|---:|---:|---|---|---|"]
    for s in seeds:
        p = rec["per_seed"][s]["primary"]["probes"]
        L.append(f"| {s.split('__')[-1]} | {p['A']['queries']:,} | {p['A']['positive_golds']:,} | {p['A']['pairs']:,} | {f3(p['A'])} | {f3(p['B'])} | {f3(p['C'])} |")
    L += ["", "Pair accuracy (unweighted): " + "; ".join(
        f"{s.split('__')[-1]} A {rec['per_seed'][s]['primary']['probes']['A']['pair_accuracy']:.3f} / B {rec['per_seed'][s]['primary']['probes']['B']['pair_accuracy']:.3f} / C {rec['per_seed'][s]['primary']['probes']['C']['pair_accuracy']:.3f}"
        for s in seeds) + ".", "",
          f"Clears the bar on all three seeds: {rec['clears_bar_all_three_seeds']}.", "",
          "## Reference orderings on the same pairs (descriptive)", "",
          "| twin seed | fixed base | twin residual | twin score | u_gnn_v2_ef |", "|---|---|---|---|---|"]
    for s in seeds:
        r = rec["per_seed"][s]["primary"]["references"]
        L.append(f"| {s.split('__')[-1]} | {f3(r['fixed_base'])} | {f3(r['twin_residual'])} | {f3(r['twin_score'])} | {f3(r['gnn_score'])} |")
    L += ["", "## Secondary (descriptive, no verdict): all G_miss vs N_disp", "",
          "| twin seed | queries | A | B | C | fixed base | twin residual | GNN |", "|---|---:|---|---|---|---|---|---|"]
    for s in seeds:
        p, r = rec["per_seed"][s]["secondary"]["probes"], rec["per_seed"][s]["secondary"]["references"]
        L.append(f"| {s.split('__')[-1]} | {p['A']['queries']:,} | {f3(p['A'])} | {f3(p['B'])} | {f3(p['C'])} | {f3(r['fixed_base'])} | {f3(r['twin_residual'])} | {f3(r['gnn_score'])} |")
    inv = {s: round(1 - rec["per_seed"][s]["primary"]["references"]["fixed_base"]["pair_auc"], 4) for s in seeds}
    L += ["", "## Selection note (read before the reading)", "",
          "The pairs are conditioned on the twin's own ranking: every negative is a candidate the twin placed at ranks 2-5 and every "
          "positive a gold it placed below 5. On such a population, features correlated with what the twin already rewards mark the "
          "negatives, so part of any probe's separation is a reversal of the twin's preference, not new evidence. The trivial reversal -- "
          "the inverted fixed base -- already reaches " + ", ".join(f"{s.split('__')[-1]} {v:.3f}" for s, v in inv.items()) +
          " on the primary pairs; probe A exceeds it by " + ", ".join(
              f"{s.split('__')[-1]} {rec['per_seed'][s]['primary']['probes']['A']['pair_auc'] - inv[s]:+.3f}" for s in seeds) + ".",
          "",
          "The probes are fitted on this conditional population only. A clear bar shows the retained columns jointly order these pairs "
          "out of fold; it does not show that one full-pool scorer can lift these golds without demoting the golds the twin already "
          "ranks well. That is a retrieval question, and a v2.1 declaration would have to test it on retrieval metrics.", ""]
    fs = rec["feature_sets"]
    p0 = rec["per_seed"][seeds[0]]["primary"]
    L += ["", "## Protocol", "",
          f"- Cross-fitting: {rec['cross_fit']['folds']} folds, fold = {rec['cross_fit']['rule']}; all pairs of a query in one fold; queries per fold on the primary task (s0): {p0['queries_per_fold']}.",
          "- Pairs: d = x_gold - x_neg entered as (d, 1) and (-d, 0); each query weighs 1 in total. Normalisation fitted on training folds only.",
          "- Probe form: per-candidate scorer g(x), pair logit g(x_gold) - g(x_neg) (for A exactly logistic regression on d without intercept).",
          f"- A: {rec['hyperparameters']['A']}; {p0['probes']['A']['parameters']} parameters.",
          f"- B, C: {rec['hyperparameters']['B_C']}; B {p0['probes']['B']['parameters']:,} / C {p0['probes']['C']['parameters']:,} parameters.",
          f"- Inputs: A/B {fs['A']['n']} (129 surviving + fixed base; sha256 `{fs['A']['sha256'][:16]}`), C {fs['C']['n']} (164 raw + fixed base; sha256 `{fs['C']['sha256'][:16]}`). "
          "No GNN score, twin score, residual, rank, arm or dataset id is an input.", "",
          "## Incidents", ""] + [f"- {i}" for i in rec["incidents"]] + ["", "## Artefacts", ""]
    L += [f"- {k}: `{v['path']}` sha256 `{v['sha256'][:16]}`" for k, v in rec["inputs"].items()]
    L += [f"- record: `outputs/umlp_d02/record.json` sha256 `{rec.get('_record_sha256', 'see run record')}`", ""]
    return LF.join(L)


def stage_doc(log=print):
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    rec["_record_sha256"] = sha256_file(RECORD)[:16]
    DOC.write_text(render_doc(rec), encoding="utf-8")
    log(f"doc {DOC}")


def stage_file(date: str, log=print):
    text = CONFIG.read_text(encoding="utf-8")
    key = f"run_record_umlp_d02_{date}"
    if key + ":" in text:
        raise SystemExit(f"{key} already filed")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    block = {key: {"utc": utc(), "record": "outputs/umlp_d02/record.json", "record_sha256": sha256_file(RECORD),
                   "document": "docs/UMLP_D02_JOINT_SEPARABILITY.md", "held_half_read": False, "gat_loaded": False,
                   "clears_bar_all_three_seeds": rec["clears_bar_all_three_seeds"], "reading": rec["reading"],
                   "status_moves": "DECLARED_NOT_RUN -> RUN", "next": "STOP with the scientific interpretation; v2.1 is its own declaration"}}
    new = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    if new == text:
        raise SystemExit("status line not found")
    CONFIG.write_text(new.rstrip(LF) + LF + LF + yaml.safe_dump(block, sort_keys=False, width=200), encoding="utf-8")
    log(f"filed {key}; status RUN")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "doc", "file"], required=True)
    ap.add_argument("--date")
    args = ap.parse_args()
    decl = load_declaration()

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
    {"run": lambda: stage_run(decl, log), "doc": lambda: stage_doc(log), "file": lambda: stage_file(args.date, log)}[args.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
