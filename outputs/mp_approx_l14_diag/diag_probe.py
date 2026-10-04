"""Descriptive look, after level 14's read (99a2685): can the query embedding alone name the true chain? Not a result,
not filed and never an arm: the qtype is a dataset label the GNN never sees, so a model fitted on it is not label-matched.
Level 12's carve sidecars and level 14's r sidecar and units are read in place, every array against its meta.json sha256,
and nothing of theirs is written.

Probes from q_emb (1536) to the qtype (49 classes, so the true chain), each fitted on level 12's fit carve (1x) or its
fit, x1, x2 and x3 carves (4x), the setting chosen by accuracy on the select carve, read once on r:

  lin      multinomial logistic regression on the standardised embedding, L2 grid
  lin64    the same through a 64-wide linear bottleneck (the typed-walk model's A is 1536 -> 64)
  mlp      1536 -> 256 -> 49, GELU, AdamW, weight-decay grid

On r: accuracy overall and by hop, the lowest qtypes, the top confusions by kind (diag_chain.py's kinds on the true
chain's tokens: shorter, longer, dir_only, rel@i, with order swaps named apart), and the cross-tab with level 14's units:
on (row, seed) pairs where a unit's posterior mode is a wrong chain while the true chain has walks in the unit's view, the
share the probe names right (if high, a better query-to-chain map has room; if low, those rows are ambiguous to the
embedding as well).

    python outputs/mp_approx_l14_diag/diag_probe.py --host    # -> diag_probe.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"

import argparse  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l14_fit as F14  # noqa: E402  (level 14's fit module, imported unchanged)

P, L0, L8, L9 = F14.P, F14.L0, F14.L8, F14.L9
SEEDS, HOPS = F14.SEEDS, F14.HOPS
OUT_JSON = HERE / "diag_probe.json"
CARVES = F14.CARVES_DIR
TRAIN = {"1x": ("fit",), "4x": ("fit", "x1", "x2", "x3")}
LIN_L2 = (1e-5, 1e-4, 1e-3, 1e-2)
MLP_WD = (1e-4, 1e-2, 1e-1)
UNITS = (("b0", "TW-1x"), ("b0", "TW-4x"), ("full", "TW-1x"))
KEYS = ("q_emb", "q_qtype", "q_hop")


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def load_part(d: Path) -> dict:
    meta = L8.read_json(d / "meta.json")
    out = {"qtypes": list(meta["qtypes"])}
    for key in KEYS:
        path = d / f"{key}.npy"
        if L0.sha256_file(path) != meta["arrays_sha256"][f"{key}.npy"]:
            raise SystemExit(f"{path}: not the array its meta.json records")
        out[key] = np.load(path)
    return out


def seq_name(seq) -> str:
    return ">".join(L8.REL_ORDER[t // 3] + ("" if t % 3 == 0 else "^-1" if t % 3 == 1 else "^~") for t in seq)


def kind(truth: tuple, seq: tuple) -> str:
    if seq == truth:
        return "same_chain"   # two qtypes whose kinds name one chain
    if len(seq) < len(truth):
        return "shorter"
    if len(seq) > len(truth):
        return "longer"
    if sorted(seq) == sorted(truth):
        return "order_swap"
    rel = [i for i in range(len(seq)) if seq[i] // 3 != truth[i] // 3]
    return "dir_only" if not rel else "rel@" + "+".join(str(i) for i in rel)


def fit_lin(X, y, Xs, ys, n_cls, l2, width=None, seed=0):
    torch.manual_seed(seed)
    d = X.shape[1]
    if width is None:
        model = torch.nn.Linear(d, n_cls)
    else:
        model = torch.nn.Sequential(torch.nn.Linear(d, width, bias=False), torch.nn.Linear(width, n_cls))
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=300, history_size=20, line_search_fn="strong_wolfe")
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(Xt), yt) + l2 * sum((p ** 2).sum() for n, p in model.named_parameters() if "bias" not in n)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = float((model(torch.as_tensor(Xs)).argmax(1).numpy() == ys).mean())
    return model, acc


def fit_mlp(X, y, Xs, ys, n_cls, wd, seed=0, epochs=40, batch=256):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, n_cls))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    for _ep in range(epochs):
        perm = torch.randperm(Xt.shape[0], generator=g)
        for s in range(0, Xt.shape[0], batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        acc = float((model(torch.as_tensor(Xs)).argmax(1).numpy() == ys).mean())
    return model, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="a laptop smoke run on the first rows of each part (no number)")
    args = ap.parse_args()
    t0 = time.time()
    P.HARD_STOP_DIR[0] = HERE   # a hard stop of this look lands here, never in level 14's outputs
    P.route_stops()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    parts = {c: load_part(CARVES / c) for c in ("select", "fit", "x1", "x2", "x3")}
    parts["r"] = load_part(F14.DATA)
    qts = parts["r"]["qtypes"]   # every part's q_qtype indexes its own list (the select carve has 48 of the 49): by name
    for c, p in parts.items():
        if not set(p["qtypes"]) <= set(qts):
            raise SystemExit(f"{c}: a qtype that r's list does not hold")
        p["q_qtype"] = np.asarray([qts.index(p["qtypes"][i]) for i in range(len(p["qtypes"]))], dtype=np.int64)[p["q_qtype"]]
    if args.limit:
        parts = {c: {k: (v[:args.limit] if isinstance(v, np.ndarray) else v) for k, v in p.items()} for c, p in parts.items()}
    n_cls = len(qts)
    truth = {i: tuple(L8.chain_tokens(L8.true_chain(qt))) for i, qt in enumerate(qts)}
    yr, hop = parts["r"]["q_qtype"].astype(np.int64), parts["r"]["q_hop"].astype(np.int64)
    out = {"look": "diag_probe", "after": "99a2685", "rows": {c: int(p["q_qtype"].size) for c, p in parts.items()},
           "classes": n_cls, "probes": {}}
    preds = {}
    for tr, names in TRAIN.items():
        X = np.concatenate([parts[c]["q_emb"] for c in names]).astype(np.float32)
        y = np.concatenate([parts[c]["q_qtype"] for c in names]).astype(np.int64)
        mu, sd = X.mean(0), X.std(0) + 1e-6

        def z(a):
            return ((a.astype(np.float32) - mu) / sd).astype(np.float32)

        Xz, Xs, ys, Xr = z(X), z(parts["select"]["q_emb"]), parts["select"]["q_qtype"].astype(np.int64), z(parts["r"]["q_emb"])
        grids = {"lin": [(l2, lambda l2=l2: fit_lin(Xz, y, Xs, ys, n_cls, l2)) for l2 in LIN_L2],
                 "lin64": [(l2, lambda l2=l2: fit_lin(Xz, y, Xs, ys, n_cls, l2, width=64)) for l2 in LIN_L2],
                 "mlp": [(wd, lambda wd=wd: fit_mlp(Xz, y, Xs, ys, n_cls, wd)) for wd in MLP_WD]}
        for probe, grid in grids.items():
            t1 = time.time()
            tried = []
            best = None
            for hp, run in grid:
                model, acc = run()
                tried.append({"setting": hp, "select_acc": acc})
                if best is None or acc > best[1]:
                    best = (model, acc, hp)
            model, sel_acc, hp = best
            with torch.no_grad():
                pr = model(torch.as_tensor(Xr)).argmax(1).numpy()
            key = f"{probe}-{tr}"
            preds[key] = pr
            right = pr == yr
            per_qt = {qts[i]: {"rows": int((yr == i).sum()), "acc": float(right[yr == i].mean())} for i in range(n_cls) if (yr == i).any()}
            conf = Counter((int(a), int(b)) for a, b in zip(yr[~right], pr[~right]))
            kinds = Counter()
            for (a, b), c in conf.items():
                kinds[kind(truth[a], truth[b])] += c
            out["probes"][key] = {"setting": hp, "grid": tried, "select_acc": sel_acc, "r_acc": float(right.mean()),
                                  "r_acc_by_hop": {f"hop={h}": float(right[hop == h].mean()) for h in HOPS},
                                  "wrong_rows": int((~right).sum()), "wrong_by_kind": dict(kinds.most_common()),
                                  "lowest_qtypes": dict(sorted(per_qt.items(), key=lambda kv: kv[1]["acc"])[:10]),
                                  "top_confusions": [{"true": qts[a], "pred": qts[b], "rows": c, "kind": kind(truth[a], truth[b]),
                                                      "true_chain": seq_name(truth[a]), "pred_chain": seq_name(truth[b])}
                                                     for (a, b), c in conf.most_common(15)],
                                  "seconds": round(time.time() - t1, 1)}
            log(f"{key:10s} setting {hp:g}: select {sel_acc:.4f}  r {right.mean():.4f}  by hop "
                + " / ".join(f"{right[hop == h].mean():.4f}" for h in HOPS) + f"  wrong by kind {dict(kinds.most_common(5))}")
    del grids, X, Xz, Xs, Xr
    for p in parts.values():
        p.pop("q_emb", None)
    # the cross-tab with level 14's units (skipped on a smoke run: the units cover all of r)
    if not args.limit:
        nb = L9.View(F14.DATA, "nb")
        present = {"b0": np.zeros(yr.size, dtype=bool), "full": np.zeros(yr.size, dtype=bool)}
        for q in range(yr.size):
            r0, r1 = L8.chain_reach(nb, q, L8.true_chain(qts[yr[q]]))
            present["b0"][q], present["full"][q] = bool(r0.size), bool(r0.size or r1.size)
        del nb
        out["units"] = {}
        for view, fit in UNITS:
            A = np.zeros((yr.size, len(SEEDS)), dtype=np.int64)
            for i, k in enumerate(SEEDS):
                npz, js = F14.unit_paths(view, fit, k)
                flog = L8.read_json(js)
                if L0.sha256_file(npz) != flog["arrays_sha256"]:
                    raise SystemExit(f"{npz}: not the arrays its log records")
                with np.load(npz) as zf:
                    if not np.array_equal(zf["q"], np.arange(yr.size)):
                        raise SystemExit(f"{npz}: not r's rows, each once")
                    A[:, i] = zf["argmax"]
            seq_right = np.array([[L8.token_sequence(int(A[q, i])) == truth[yr[q]] for i in range(len(SEEDS))] for q in range(yr.size)])
            is_null = A < 0
            wrong_present = ~seq_right & ~is_null & present[view][:, None]
            res = {"chain_right": float(seq_right.mean()), "chain_right_by_hop": {f"hop={h}": float(seq_right[hop == h].mean()) for h in HOPS},
                   "chain_right_given_present": float(seq_right[present[view]].mean()), "wrong_present_pairs": int(wrong_present.sum()),
                   "probe_right_on_wrong_present": {}}
            for key, pr in preds.items():
                pr_right = np.repeat((pr == yr)[:, None], len(SEEDS), 1)
                res["probe_right_on_wrong_present"][key] = float(pr_right[wrong_present].mean()) if wrong_present.any() else None
            out["units"][f"{view}/{fit}"] = res
            log(f"{view}/{fit}: chain right {res['chain_right']:.4f} (given present {res['chain_right_given_present']:.4f}); "
                f"wrong-but-present pairs {res['wrong_present_pairs']}; probe right there: "
                + ", ".join(f"{k} {v:.3f}" for k, v in res["probe_right_on_wrong_present"].items() if v is not None))
    out.update({"limit": args.limit, "diag_script_sha256": L0.sha256_file(Path(__file__)), "seconds": round(time.time() - t0, 1)})
    L8.write_json(OUT_JSON if not args.limit else HERE / "diag_probe.smoke.json", out)
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
