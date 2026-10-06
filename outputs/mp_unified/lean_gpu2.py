"""Step 2 trainer (docs/STEP2_EVAL_MIX_WEIGHTS.md, section 3): lean_gpu's trainer with each fit question's listwise
term multiplied by its weight (outputs/step2/weights.json) before the step's mean over its questions with gold. All
else is lean_gpu's: `train` runs lean_gpu.train_cmd with lean_gpu's fit loop replaced by fit_variant_w (lean_gpu's
fit_variant line for line, the loss line weighted) and its cache carve by one that carries the weights. Reads and the
CPU read check are lean_gpu's own commands, run with --out-root outputs/step2/fits.

    python outputs/mp_unified/lean_gpu2.py train --fit J5 --device cuda --host
    python outputs/mp_unified/lean_gpu2.py identity --cache-root DIR [--identity-out FILE]   # CPU, laptop

Checks:
  weights   weights.json has the declared sha256; a carve's weights are weights.json's for its dataset, whose
            ids_sha256 is the carve's carve_ids_sha256 and whose ids cover the cache's ids (matched by question id).
            train.json records weights.json's sha256, this script's sha256 and each carve's weight summary.
  identity  with every weight 1, fit_variant_w's states equal lean_gpu.fit_variant's bit for bit (every epoch and the
            SWA state, p and pf, two epochs on the 2wiki fit carve, CPU); with the filed weights they differ.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_gpu as LG  # noqa: E402

LC, L8 = LG.LC, LG.L8
OUT = ROOT / "outputs" / "step2" / "fits"
WEIGHTS = ROOT / "outputs" / "step2" / "weights.json"
WEIGHTS_SHA256 = "90d263db6de7d4a921df0db5851abc490a5db0639d9412797dde1cf2d595d564"   # the declaration, section 2
log = LG.log


def load_weights(path=WEIGHTS):
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != WEIGHTS_SHA256:
        raise SystemExit(f"{path}: sha256 {got} is not the declared {WEIGHTS_SHA256}")
    return json.loads(raw.decode("utf-8")), got


def listwise_w(scores, gold, nq, B, w):
    """lean_gpu.listwiseD with each question's term times its weight w (B,), before the mean over questions with gold."""
    lp = LG.seg_log_softmaxD(scores, nq, B)
    g = gold.to(scores.dtype)
    ng = torch.zeros(B, device=scores.device).index_add_(0, nq, g)
    pq = -torch.zeros(B, device=scores.device).index_add_(0, nq, lp * g)
    has = ng > 0
    return (w[has] * (pq[has] / ng[has])).mean() if bool(has.any()) else scores.sum() * 0.0


def attach(c, wrec, unit=False):
    """The carve's per-question weights (float32 on its device), from weights.json's entry for its dataset."""
    if unit:
        c.w = torch.ones(c.rows, dtype=torch.float32, device=c.device)
        c.w_summary = {"unit": True}
        return c
    ent = wrec["per_dataset"].get(c.ds)
    if ent is None or c.carve != LG.FIT_CARVE[c.ds] or c.look["carve_ids_sha256"] != ent["ids_sha256"]:
        raise SystemExit(f"{c.ds}/{c.carve}: no weights.json entry for this carve")
    of = dict(zip(ent["ids"], ent["weights"]))
    miss = [q for q in c.ids if q not in of]
    if miss:
        raise SystemExit(f"{c.ds}/{c.carve}: {len(miss)} questions without a weight, e.g. {miss[:3]}")
    w = np.asarray([of[q] for q in c.ids], np.float64)
    c.w = torch.from_numpy(w.astype(np.float32)).to(c.device)
    c.w_summary = {"questions": int(w.size), "mean": float(w.mean()), "min": float(w.min()), "max": float(w.max()),
                   "weight_of_stratum": ent["weight_of_stratum"]}
    return c


def fit_variant_w(tr, blocks, cfg, seed, hidden, ctx, device, tag=""):
    """lean_gpu.fit_variant with the loss weighted per question (the carve's w); every other line as there."""
    if cfg["cos"] or cfg["adamw"] or cfg["drop"]:
        raise SystemExit("fit8 takes lr:wd:dropout:epochs:swa_from only")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: tr[0].widths[b] for b in blocks}
    model = LG.LeanMLP8D(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed, arm="ctl", ctx=ctx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    t_ctx = time.time()
    stats = LG.set_ctx_statsD(model, tr, blocks)
    if stats:
        stats["seconds"] = time.time() - t_ctx
    units_ci = np.concatenate([np.full(c.rows, ci, np.int64) for ci, c in enumerate(tr)])
    units_q = np.concatenate([np.arange(c.rows, dtype=np.int64) for c in tr])
    keep_all = None
    states, curve = [], []
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(units_ci.size)
        t0 = time.time()
        tot = torch.zeros((), dtype=torch.float64, device=device)
        nb, steps = 0, 0
        guard = {"skipped_loss": 0, "skipped_grad": 0, "first": None}
        for k in range(0, order.size, 32):
            sel = order[k:k + 32]
            cis, qq = units_ci[sel], units_q[sel]
            for ci in sorted(set(cis.tolist())):
                qs = qq[cis == ci]
                feats, nq, base_z, gold = tr[ci].batch(qs, blocks)
                B = qs.size
                steps += 1
                if keep_all is None or keep_all.shape[0] != B:
                    keep_all = torch.ones((B, len(blocks)), dtype=torch.float32, device=device)
                s = model(feats, keep_all, nq, B, base_z)
                loss = listwise_w(s, gold, nq, B, tr[ci].w[torch.from_numpy(qs).to(tr[ci].device)])
                if not bool(torch.isfinite(loss).all()):
                    guard["skipped_loss"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "loss", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]]}
                    continue
                opt.zero_grad()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                    guard["skipped_grad"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "grad", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]],
                                          "nonfinite_grads": [n_ for n_, p in model.named_parameters()
                                                              if p.grad is not None and not bool(torch.isfinite(p.grad).all())][:20]}
                    opt.zero_grad()
                    continue
                opt.step()
                tot += loss.detach().to(torch.float64)
                nb += 1
        states.append(LG.cpu_state(model))
        rec = {"epoch": ep, "loss": float(tot) / max(nb, 1), "steps": steps, "seconds": time.time() - t0}
        if ctx != "none":
            rec["gate_w_norm"] = float(model.gate_w.detach().norm())
            rec["gate_b"] = float(model.gate_b.detach())
        if guard["skipped_loss"] or guard["skipped_grad"]:
            rec["guard"] = guard
            log(f"  {tag} ep {ep}: GUARD skipped {guard['skipped_loss']} at a non-finite loss, {guard['skipped_grad']} at a "
                f"non-finite gradient; first {json.dumps(guard['first'])}")
        curve.append(rec)
        log(f"  {tag} ep {ep}: loss {rec['loss']:.4f}, {steps} steps ({rec['seconds']:.0f}s)")
    acc, n_acc = None, 0
    for ep, st in enumerate(states):
        if ep < cfg["swa_from"]:
            continue
        if acc is None:
            acc = {k: v.double().clone() for k, v in st.items() if k not in L8.CTX_BUFFERS}
        else:
            for k in acc:
                acc[k] += st[k].double()
        n_acc += 1
    swa = {k: ((acc[k] / n_acc).to(torch.float32) if k in acc else states[-1][k].clone()) for k in states[-1]}
    return {"states": states, "swa": swa, "curve": curve, "widths": widths, "ctx_stats": stats,
            "swa_epochs": list(range(cfg["swa_from"], cfg["epochs"]))}


def weighted_carve_class(wrec, unit=False):
    class WCarve(LG.CacheCarve):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            attach(self, wrec, unit)
    return WCarve


def train_cmd(a):
    wrec, wsha = load_weights(a.weights)
    made = []

    class Recording(weighted_carve_class(wrec)):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            made.append(self)
    LG.CacheCarve, LG.fit_variant, LG.OUT = Recording, fit_variant_w, OUT
    rc = LG.train_cmd(a)
    name = a.fit or a.name
    fdir = Path(a.out_root or OUT) / name
    rec = json.loads((fdir / "train.json").read_text(encoding="utf-8"))
    rec["step2"] = {"declared_in": "docs/STEP2_EVAL_MIX_WEIGHTS.md", "trainer": "outputs/mp_unified/lean_gpu2.py",
                    "trainer_sha256": LC.sha_src(__file__), "weights_sha256": wsha,
                    "weights": {f"{c.ds}={c.carve}": c.w_summary for c in made}}
    LC.write_json(fdir / "train.json", rec)
    log(f"train {name}: weights {wsha[:12]} " + ", ".join(f"{c.ds} mean {c.w_summary['mean']:.4f}" for c in made))
    return rc


def identity_cmd(a):
    """CPU: unit weights reproduce lean_gpu.fit_variant bit for bit; the filed weights change the states."""
    t0 = time.time()
    torch.set_num_threads(a.threads)
    flags = LG.set_flags("cpu")
    LG.bind_device_ops()
    wrec, wsha = load_weights(a.weights)
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    cfg = LG.L5.parse_configs("x=" + a.config)["x"]
    variants = [v for v in LG.VARIANTS if v[0] in a.variants.split(",")]
    tr = [LG.CacheCarve("2wiki", "fit", "2wiki", "cpu", cache_root)]
    used = sorted({b for v in variants for b in LG.SETS[v[1]] if b not in LG.LM.LEAN})
    dead = LG.dead_blocks(tr, used)
    out = {"weights_sha256": wsha, "config": cfg, "flags": flags, "dead": dead, "variants": {}}
    ok = True
    for vname, set_, ctx in variants:
        bl = [b for b in LG.SETS[set_] if b not in dead]
        ref = LG.fit_variant(tr, bl, cfg, a.seed, a.hidden, ctx, "cpu", tag=f"lean_gpu/{vname}")
        attach(tr[0], wrec, unit=True)
        one = fit_variant_w(tr, bl, cfg, a.seed, a.hidden, ctx, "cpu", tag=f"unit/{vname}")
        bad = sorted({k for e in range(len(ref["states"])) for k in LG.same_state(ref["states"][e], one["states"][e])}
                     | set(LG.same_state(ref["swa"], one["swa"])))
        attach(tr[0], wrec)
        wtd = fit_variant_w(tr, bl, cfg, a.seed, a.hidden, ctx, "cpu", tag=f"weighted/{vname}")
        moved = bool(LG.same_state(ref["states"][-1], wtd["states"][-1]))
        ok = ok and not bad and moved
        out["variants"][vname] = {"unit_identical": not bad, "differing": bad, "weighted_differs": moved,
                                  "loss": {"lean_gpu": [r["loss"] for r in ref["curve"]],
                                           "unit": [r["loss"] for r in one["curve"]],
                                           "weighted": [r["loss"] for r in wtd["curve"]]}}
        log(f"  identity {vname}: unit weights {'IDENTICAL' if not bad else 'DIFFERENT ' + str(bad)}; "
            f"filed weights {'change' if moved else 'do NOT change'} the states")
    out["verdict"] = "IDENTICAL" if ok else "FAIL"
    out["seconds"] = time.time() - t0
    out["script_sha256"] = LC.sha_src(__file__)
    out["lean_gpu_sha256"] = LC.sha_src(LG.__file__)
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if a.identity_out:
        LC.write_json(a.identity_out, out)
    log(f"identity: {out['verdict']} ({out['seconds']:.0f}s)")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "identity"))
    ap.add_argument("--fit", choices=sorted(LG.FITS))
    ap.add_argument("--name", help="a run name outside FITS (with --train and --basis)")
    ap.add_argument("--train", default="", help="DS=CARVE,... (without --fit)")
    ap.add_argument("--basis", default="2wiki")
    ap.add_argument("--variants", default="p,pf,n,nf")
    ap.add_argument("--config", default=LG.CONFIG)
    ap.add_argument("--seed", type=int, default=LG.SEED)
    ap.add_argument("--hidden", type=int, default=LG.HIDDEN)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--cache-root")
    ap.add_argument("--out-root")
    ap.add_argument("--identity-out")
    ap.add_argument("--no-verify", action="store_true", help="skip the cache arrays' sha256 check on load")
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        torch.set_num_threads(a.threads)
        if not a.fit and not (a.name and a.train):
            raise SystemExit("train: --fit, or --name with --train and --basis")
        return train_cmd(a)
    if a.config == LG.CONFIG:
        a.config = "2e-3:1e-4:0.1:2:1"
    a.variants = "p,pf" if a.variants == "p,pf,n,nf" else a.variants
    return identity_cmd(a)


if __name__ == "__main__":
    sys.exit(main())
