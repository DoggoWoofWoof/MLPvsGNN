"""lean_mlp7 with a guard against non-finite steps; everything else is lean_mlp7's, imported and called unchanged.

l7-j3a (lean_mlp7, trained on 2wiki x4, hotpot's fit carve and metaqa's fit carve together) lost its ctl arm at epoch
4 of 8: the epoch's loss was nan and its select R@5 fell from 0.8195 to 0.0366. One batch's loss or gradient was not
finite, Adam took the step, and from then on every weight was nan, so the weight average (from epoch 2) and the last
weights were lost too; only the select-chosen best (epoch 3) survived. The joint run was cancelled with its sibling
l7-j3b and both are rerun under this file.

fit7g is lean_mlp7.fit7 line for line with two checks per batch:
  - a batch whose loss (the listwise loss plus the arm's extra term) is not finite takes no step;
  - a batch whose gradients are not all finite takes no step either (its gradients are cleared).
Each skipped batch is counted. The epoch's curve entry gets a "guard" record ({skipped_loss, skipped_grad, first}),
and so does the log, only in an epoch where a batch was skipped. "first" describes the epoch's first skipped batch:
its graph, its rows, the input blocks carrying non-finite values, the count of non-finite scores, the largest finite
score, and the parameters with non-finite gradients.
EMState.observe skips a batch whose prior or posterior is not finite, so theta's counts stay finite. It counts the skip
on the state (em_skipped). Such a batch's loss is not finite either, so fit7g takes no step on it.

With no non-finite batch the checks only read values, so the fit is lean_mlp7's bit for bit: every arm's best, swa
and last weights, its best epoch and its curve. The selftest checks this on lean_mlp7's toy carves for all six arms.
It also reproduces lean_mlp7's own failure on the toy carves: at lean_mlp7's selftest rate (3e-2), arm aw meets a
batch at epoch 3 whose loss and scores are finite but whose AW-net gradients (P, Wq, dirv, bias, other) are not. Then it
puts a nan into one batch's input. In both cases the unguarded fits end with non-finite weights; the guarded fits skip
that one batch, keep every weight finite and record the skip.

    python outputs/mp_unified/lean_host7g.py lean_mlp7g ARGS...    (lean_mlp7's own arguments)
    python outputs/mp_unified/lean_mlp7g.py --selftest
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_mlp7 as L7  # noqa: E402

LM, L3, L4, L5 = L7.LM, L7.L3, L7.L4, L7.L5
_OBSERVE = L7.EMState.observe


def _finite(t):
    return bool(torch.isfinite(t).all())


def observe_g(self, g, labs, q_has, lp, q, o):
    """EMState.observe, skipping a batch whose prior or posterior is not finite."""
    if not (_finite(lp) and _finite(q)):
        self.em_skipped = getattr(self, "em_skipped", 0) + 1
        return None
    return _OBSERVE(self, g, labs, q_has, lp, q, o)


def _bad_inputs(feats):
    bad = []
    for b, v in feats.items():
        items = v.items() if isinstance(v, dict) else [("", v)]
        for k, x in items:
            if torch.is_tensor(x) and x.is_floating_point() and not _finite(x):
                bad.append(f"{b}.{k}" if k else b)
            elif isinstance(x, np.ndarray) and x.dtype.kind == "f" and not np.isfinite(x).all():
                bad.append(f"{b}.{k}" if k else b)
    return bad


def describe(carve, qs, feats, s, loss, x, model=None):
    fin = torch.isfinite(s.detach())
    d = {"graph": getattr(carve, "ds", None), "rows": [int(v) for v in qs[:32]], "n_rows": int(qs.size),
         "nonfinite_inputs": _bad_inputs(feats), "scores_nonfinite": int((~fin).sum()),
         "max_abs_finite_score": float(s.detach()[fin].abs().max()) if bool(fin.any()) else None,
         "loss": None if loss is None else float(loss.detach()), "extra": None if x is None else float(x.detach())}
    if model is not None:
        d["nonfinite_grads"] = [n for n, p in model.named_parameters() if p.grad is not None and not _finite(p.grad)][:20]
    return d


def fit7g(train, select, blocks, arm, cfg, seed, hidden, emc=None, aux_omega=1.0):
    """lean_mlp7.fit7 with the step guard (see the module docstring)."""
    if cfg["cos"] or cfg["adamw"] or cfg["drop"]:
        raise SystemExit("fit7 takes lr:wd:dropout:epochs:swa_from only")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: train[0].widths[b] for b in blocks}
    model = L7.LeanMLP7(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed, arm=arm)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    units = [(ci, q) for ci, c in enumerate(train) for q in range(c.rows)]
    aw = "AW" in blocks
    em = None
    if aw and arm in L7.EM_ARMS:
        em = L7.EMState(L7.LeanMLP7.cfg.get("k_types", 8), sorted({c.ds for c in train}), emc["lam0"] if arm == "em" else 0.0,
                        emc["an"], emc["mu"], emc["omega"])
    keep_all = None
    best, best_state, best_ep, curve = -1.0, None, -1, []
    acc, n_acc = None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(units))
        t0 = time.time()
        tot, nb, tot_x, nx = 0.0, 0, 0.0, 0
        lam = em.lam(ep) if em is not None else 0.0
        guard = {"skipped_loss": 0, "skipped_grad": 0, "first": None}
        for k in range(0, len(order), 32):
            chunk = [units[j] for j in order[k:k + 32]]
            for ci in sorted({c for c, _ in chunk}):
                qs = np.asarray([q for c, q in chunk if c == ci])
                feats, nq, base_z, gold, _ = L7.batch_of7(train[ci], qs, blocks)
                B = qs.size
                if keep_all is None or keep_all.shape[0] != B:
                    keep_all = torch.ones((B, len(blocks)), dtype=torch.float32)
                s = model(feats, keep_all, nq, B, base_z)
                loss = LM.listwise(s, gold, nq, B)
                if not _finite(loss):
                    guard["skipped_loss"] += 1
                    if guard["first"] is None:
                        guard["first"] = dict(describe(train[ci], qs, feats, s, loss, None), epoch=ep, at="loss")
                    continue
                x = None
                if aw and arm == "aux":
                    x = model.awn.aux_loss(feats["AW"])
                    x = None if x is None else aux_omega * x
                elif em is not None:
                    x = model.awn.em_loss(feats["AW"], em, lam)
                if x is not None and not _finite(x):
                    guard["skipped_loss"] += 1
                    if guard["first"] is None:
                        guard["first"] = dict(describe(train[ci], qs, feats, s, loss, x), epoch=ep, at="extra")
                    continue
                if x is not None:
                    tot_x += float(x.item())
                    nx += 1
                    loss = loss + x
                opt.zero_grad()
                loss.backward()
                if not all(_finite(p.grad) for p in model.parameters() if p.grad is not None):
                    guard["skipped_grad"] += 1
                    if guard["first"] is None:
                        guard["first"] = dict(describe(train[ci], qs, feats, s, loss, x, model), epoch=ep, at="grad")
                    opt.zero_grad()
                    continue
                opt.step()
                tot += loss.item()
                nb += 1
        q = LM.quality(model, select, blocks, {b: 1.0 for b in blocks})
        state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "extra": tot_x / max(nx, 1) if nx else None, "select": q,
               "seconds": time.time() - t0}
        if aw and model.awn.kappa is not None:
            rec["kappa"] = [round(float(v), 4) for v in model.awn.kappa.detach()]
        if em is not None:
            rec["em"] = em.epoch_trace(ep, lam)
            rec["eta"] = [round(float(v), 4) for v in model.awn.eta.detach()]
        if guard["skipped_loss"] or guard["skipped_grad"]:
            if em is not None:
                guard["em_skipped_total"] = getattr(em, "em_skipped", 0)
            rec["guard"] = guard
        curve.append(rec)
        L7.log(f"  {arm} ep {ep}: loss {rec['loss']:.4f} extra {rec['extra']} select R@5/FC@5/hit@1 {[round(v, 4) for v in q]} "
               f"({rec['seconds']:.0f}s){' kappa ' + str(rec['kappa']) if 'kappa' in rec else ''}"
               f"{' em ' + json.dumps(rec['em']) + ' eta ' + str(rec['eta']) if 'em' in rec else ''}")
        if "guard" in rec:
            L7.log(f"  {arm} ep {ep}: GUARD skipped {guard['skipped_loss']} batch(es) at a non-finite loss and "
                   f"{guard['skipped_grad']} at a non-finite gradient; first {json.dumps(guard['first'])}")
        score = 0.5 * (q[0] + q[1])
        if score > best:
            best, best_state, best_ep = score, state, ep
        if ep >= cfg["swa_from"]:
            if acc is None:
                acc = {k_: v.double().clone() for k_, v in state.items()}
            else:
                for k_, v in state.items():
                    acc[k_] += v.double()
            n_acc += 1
    swa_state = {k_: (v / n_acc).to(torch.float32) for k_, v in acc.items()}
    last_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    return {"best": best_state, "swa": swa_state, "last": last_state, "best_epoch": best_ep, "curve": curve, "widths": widths,
            "swa_epochs": list(range(cfg["swa_from"], cfg["epochs"]))}


def install():
    L7.fit7 = fit7g
    L7.EMState.observe = observe_g


def _same(f, g):
    for kind in ("best", "swa", "last"):
        assert set(f[kind]) == set(g[kind]), kind
        for k, v in f[kind].items():
            assert torch.equal(v, g[kind][k]), (kind, k)
    assert f["best_epoch"] == g["best_epoch"] and f["swa_epochs"] == g["swa_epochs"]
    strip = (lambda c: [{k: v for k, v in r.items() if k != "seconds"} for r in c])
    assert json.dumps(strip(f["curve"]), sort_keys=True) == json.dumps(strip(g["curve"]), sort_keys=True)


def _all_finite(f):
    return all(_finite(v) for kind in ("best", "swa", "last") for v in f[kind].values() if v.is_floating_point())


def selftest():
    torch.set_num_threads(1)
    fit7_orig, observe_orig = L7.fit7, L7.EMState.observe
    L3.LeanMLP3.dim = LM.PROJ_DIM
    L4.LeanMLP4.cfg = {"r": 8, "D": 16, "k_res": 0, "p_drop": 0.25}
    L7.LeanMLP7.cfg = {"r": 8, "D": 16, "k_res": 0, "p_drop": 0.25, "k_types": 4}
    old_batch, old_b7 = LM.batch_of, L7.batch_of7
    LM.batch_of = L7.batch_of7
    base = L5.parse_configs("b=1e-2:1e-4:0.1:4:1")["b"]     # every unguarded toy arm stays finite at this rate
    hot = L5.parse_configs("b=3e-2:1e-4:0.1:4:1")["b"]      # lean_mlp7's own selftest rate: aw meets a nan gradient
    emc = {"lam0": 1.0, "an": 2, "mu": 1.0, "omega": 1.0}
    blocks_of = {"ctl": ["rank", "WALK"]}
    try:
        # 1. no non-finite batch: fit7g = fit7 bit for bit on every arm (and fit7 repeats itself)
        for arm in L7.ARMS:
            bl = blocks_of.get(arm, ["rank", "WALK", "AW"])
            tr, se = [L7.ToyCarve7(60, 1), L7.ToyCarve7(40, 4)], [L7.ToyCarve7(30, 2)]
            f = fit7_orig(tr, se, bl, arm, base, 0, 32, emc if arm in L7.EM_ARMS else None)
            assert _all_finite(f), f"the toy {arm} fit must stay finite at this rate for the identity check"
            f2 = fit7_orig(tr, se, bl, arm, base, 0, 32, emc if arm in L7.EM_ARMS else None)
            _same(f, f2)
            L7.EMState.observe = observe_g
            g = fit7g(tr, se, bl, arm, base, 0, 32, emc if arm in L7.EM_ARMS else None)
            L7.EMState.observe = observe_orig
            _same(f, g)
            assert all("guard" not in r for r in g["curve"]) and _all_finite(g)
        # 2. lean_mlp7's own failure: at 3e-2 the toy aw arm's loss and scores stay finite but the AW net's gradients do
        #    not (epoch 3); unguarded every later weight is nan, guarded that one batch is skipped
        tr, se = [L7.ToyCarve7(60, 1), L7.ToyCarve7(40, 4)], [L7.ToyCarve7(30, 2)]
        f = fit7_orig(tr, se, ["rank", "WALK", "AW"], "aw", hot, 0, 32)
        assert not _all_finite(f)
        g = fit7g(tr, se, ["rank", "WALK", "AW"], "aw", hot, 0, 32)
        gs = [r["guard"] for r in g["curve"] if "guard" in r]
        assert _all_finite(g) and len(gs) == 1 and gs[0]["skipped_grad"] == 1 and gs[0]["skipped_loss"] == 0, gs
        fi = gs[0]["first"]
        assert fi["at"] == "grad" and not fi["nonfinite_inputs"] and fi["scores_nonfinite"] == 0, fi
        assert fi["nonfinite_grads"] and all(n.startswith("awn.") for n in fi["nonfinite_grads"]), fi
        # 3. one batch carries a nan input: unguarded the weight average and the last weights are lost; guarded not
        calls = {"n": 0}

        def poisoned(carve, qs, blocks):
            feats, nq, base_z, gold, idx = old_b7(carve, qs, blocks)
            calls["n"] += 1
            if calls["n"] == 7:
                feats = dict(feats)
                feats["rank"] = feats["rank"].clone()
                feats["rank"].view(-1)[0] = float("nan")
            return feats, nq, base_z, gold, idx

        for arm in L7.ARMS:
            bl = blocks_of.get(arm, ["rank", "WALK", "AW"])
            tr, se = [L7.ToyCarve7(60, 1), L7.ToyCarve7(40, 4)], [L7.ToyCarve7(30, 2)]
            L7.batch_of7 = poisoned
            calls["n"] = 0
            f = fit7_orig(tr, se, bl, arm, base, 0, 32, emc if arm in L7.EM_ARMS else None)
            assert not _all_finite(f), f"the unguarded {arm} fit should have lost its weights"
            calls["n"] = 0
            L7.EMState.observe = observe_g
            g = fit7g(tr, se, bl, arm, base, 0, 32, emc if arm in L7.EM_ARMS else None)
            L7.EMState.observe = observe_orig
            L7.batch_of7 = old_b7
            assert _all_finite(g), f"the guarded {arm} fit kept a non-finite weight"
            gs = [r["guard"] for r in g["curve"] if "guard" in r]
            assert len(gs) == 1 and gs[0]["skipped_loss"] + gs[0]["skipped_grad"] == 1, gs
            assert gs[0]["first"]["nonfinite_inputs"] == ["rank"] and gs[0]["first"]["epoch"] == 0, gs[0]["first"]
            if arm in L7.EM_ARMS:
                assert all(np.isfinite(r["em"]["H_post"]) for r in g["curve"])
        # 4. observe_g leaves theta's counts alone on a non-finite posterior and counts the skip
        em = L7.EMState(4, ["toy0"], 1.0, 2)
        lp = torch.log_softmax(torch.randn(5, 4), 1)
        q = torch.softmax(torch.randn(5, 4), 1)
        labs = torch.tensor([0, 1, 2, 0, 1])
        qn = q.clone()
        qn[2, 1] = float("nan")
        observe_g(em, "toy0", labs, qn, lp, qn, torch.ones(5, dtype=torch.bool))
        assert float(em.C["toy0"].sum()) == 0.0 and em.em_skipped == 1 and em.tr["units"] == 0
        observe_g(em, "toy0", labs, q, lp, q, torch.ones(5, dtype=torch.bool))
        assert abs(float(em.C["toy0"].sum()) - 5.0) < 1e-5 and em.tr["units"] == 5     # float32 rows sum to 1 within ~1e-7
        # 5. install() binds fit7g and the observe guard into lean_mlp7, which main calls by name
        install()
        assert L7.fit7 is fit7g and L7.EMState.observe is observe_g
    finally:
        L7.fit7, L7.EMState.observe = fit7_orig, observe_orig
        LM.batch_of, L7.batch_of7 = old_batch, old_b7
    print("selftest: fit7g = lean_mlp7.fit7 bit for bit on all six arms without a non-finite batch; lean_mlp7's own "
          "nan (finite loss, non-finite AW gradients) and a nan input are each skipped as one batch, every weight stays "
          "finite and the skip is recorded, where the unguarded fits lose their weights. all checks passed")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    install()
    L7.main()
    out = None
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    if out is not None and out.exists():
        res = json.loads(out.read_text(encoding="utf-8"))
        res["nan_guard"] = {"file": "outputs/mp_unified/lean_mlp7g.py",
                            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        tmp.replace(out)


if __name__ == "__main__":
    main()
