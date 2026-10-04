"""Design look (untracked; not a result and not filed): anchor-typed walks, round 8: the bonus model taught by the GNN's
scores as well as by the golds (distillation).

Rounds 2 to 7 fit the bonus model on the golds alone: a row whose golds the twin already ranks well teaches little,
and a gold-only fit reads its best on the select rows by epoch 1 to 3 and then falls (round 5). The single-type oracle
(diag_oracle.py: 1.61 / 1.40 on A256-1) says the reach sets hold what the GNN lifts; what is missing is p(tau | q).
Every look row already stores the GNN's own score for every pool node (l16_look_score.py's score columns: twin0..2,
gnn0..2), so each training row can also say where the GNN puts its mass. The loss is
    L = ce * CE(golds | s) + kd * T^2 * CE(softmax(z_T / T) | softmax(s / T))
over the row's pool, s = z_twin + kappa * bonus as before; z_T is the teacher: g0, the z-scored score of GNN seed 0 (the
model every rho is read against), or gm, the z-score of the mean of the three GNN seeds' z-scores. Under rule p the
protected node leaves the student's and the teacher's softmax as it leaves the golds. With ce 0 the fit reads no gold at
all (only the selection on the select rows does, as in every round). Nothing is read at scoring time but what round 7
reads: the teacher enters the loss only, so the model, its inputs and its non-message-passing boundary are unchanged.

The model, the bases, the variant syntax, the optimiser (Adam at 3e-3 and no decay unless --lr/--wd), the batches,
the selection and the reads are anchor_walk7.py's, imported unchanged. Records add the teacher's own read on B and x1.

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk8.py --dataset 2wiki --variants x4+x5+x6:A256-1 --kd 1 [--ce 1]
        [--T 1] [--teacher gm] [--out PATH]
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_walk7 as AW7  # noqa: E402

AW6, AW5, AW4, AW3 = AW7.AW6, AW7.AW5, AW7.AW4, AW7.AW3
AW, G2, A16 = AW7.AW, AW7.G2, AW7.A16
AW7_SHA = "e153df784112dd9dd66130dbd86a8ac7357956d94be938d2dc30c4876d22f969"
TEACHERS = ("g0", "gm")
GNN_COLS = (3, 4, 5)
log = A16.log


def teacher_of(q, kind):
    sc = np.asarray(q["score"], dtype=np.float64)
    if kind == "g0":
        return A16.zscore(sc[:, GNN_COLS[0]]).astype(np.float32)
    return A16.zscore(np.mean([A16.zscore(sc[:, c]) for c in GNN_COLS], 0)).astype(np.float32)


def pad_teacher(rows, Q, zT_of, N):
    zT = torch.full((len(rows), N), float("-inf"))
    for bi, i in enumerate(rows):
        zT[bi, :Q[i]["n"]] = torch.as_tensor(zT_of[i])
    return zT


def fit_read_kd(Q, TY, nt, tr_rows, val_rows, read_sets, z_of, zT_of, make, epochs, seed, rule, lr, wd, ce, kd, T):
    """anchor_walk7.fit_read_opt's own loop (the same draws, order, batches, optimiser, selection and reads) with the
    loss above; at kd 0 and ce 1 it is that loop."""
    torch.manual_seed(seed)
    model = make()
    mats = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    rng = np.random.default_rng(seed)
    tr = list(tr_rows)
    best, best_state, best_ep, curve = -1.0, None, -1, []
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        model.train()
        for s0 in range(0, len(tr), 128):
            rows = [tr[j] for j in perm[s0:s0 + 128]]
            PX = G2.LG.pack_gate(rows, Q, TY, z_of, nt)
            s, gold = G2.scores(model, PX)
            zT = pad_teacher(rows, Q, zT_of, s.shape[1]) if kd > 0 else None
            if rule == "p":
                first = G2.top1(PX[0][9])
                ar = torch.arange(len(rows))
                s = s.clone()
                s[ar, first] = float("-inf")
                gold = gold.clone()
                gold[ar, first] = 0.0
                if zT is not None:
                    zT[ar, first] = float("-inf")
            loss = None
            if ce > 0:
                keep = gold.sum(1) > 0
                if bool(keep.any()):
                    ls = torch.log_softmax(s[keep], 1)
                    gk = gold[keep]
                    g = gk / gk.sum(1, keepdim=True)
                    loss = ce * -(torch.where(gk > 0, ls, torch.zeros_like(ls)) * g).sum(1).mean()
            if kd > 0:
                lpS = torch.log_softmax(s / T, 1)
                lpS = torch.where(torch.isfinite(lpS), lpS, torch.zeros_like(lpS))
                pT = torch.softmax(zT / T, 1)
                lk = kd * T * T * -(pT * lpS).sum(1).mean()
                loss = lk if loss is None else loss + lk
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
        if rule == "p":
            score = float(G2.read_rows(model, Q, TY, nt, list(val_rows), z_of)[:, :2].mean())
        else:
            score = float(AW6.read_rule(model, Q, TY, nt, list(val_rows), z_of, None).mean())
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    margin, by_margin = (0.0 if rule == "p" else None), None
    if rule == "mp":
        by_margin, best_v = {}, -1.0
        for mg in AW6.MARGINS:
            v = float(AW6.read_rule(model, Q, TY, nt, list(val_rows), z_of, mg).mean())
            by_margin[str(mg)] = round(v, 4)
            if v > best_v + 1e-12:
                margin, best_v = mg, v
    reads = {name: AW6.read_rule(model, Q, TY, nt, list(rows), z_of, margin) for name, rows in read_sets.items()}
    return model, reads, best_ep, curve, margin, by_margin


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--variants", default="x4+x5+x6:A256-1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=AW3.EPOCHS)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ce", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--teacher", default="gm", choices=TEACHERS)
    a = ap.parse_args(argv)
    if a.ce < 0 or a.kd < 0 or a.ce + a.kd <= 0 or a.T <= 0:
        raise SystemExit("ce and kd must be non-negative, not both 0, and T positive")
    if AW.sha(Path(AW7.__file__)) != AW7_SHA:
        raise SystemExit("anchor_walk7.py is not the pinned file")
    if AW.sha(Path(AW6.__file__)) != AW7.AW6_SHA:
        raise SystemExit("anchor_walk6.py is not the pinned file")
    for mod, want in ((AW3, AW6.PINS["anchor_walk3"]), (AW4, AW6.PINS["anchor_walk4"]), (AW5, AW6.PINS["anchor_walk5"])):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    for mod, want in ((AW, AW3.AW_SHA), (G2, AW3.G2_SHA), (G2.LG, AW3.G1_SHA), (A16, AW.A16_SHA)):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    AW6.rebind(a.dataset)
    out_path = Path(a.out) if a.out else HERE / f"anchor_walk8_{a.dataset}.json"
    torch.set_num_threads(2)
    t0 = time.time()
    train_looks = AW6.TRAIN[a.dataset]
    specs = {name: AW7.parse(name, train_looks) for name in a.variants.split(",")}
    phi, phi_sha = None, None
    if any(s["model"].removesuffix("sc") in AW7.TXT for s in specs.values()):
        pth, want = AW7.PHRASE[a.dataset]
        phi_sha = AW.sha(pth)
        if want is None or phi_sha != want:
            raise SystemExit(f"{pth} is not the pinned phrase table")
        phi = np.load(pth).astype(np.float32)
        if phi.shape != (AW7.PHRASE_K, 1536):
            raise SystemExit("unexpected phrase table shape")
    looks = ["x1", "select"] + [lk for lk in train_looks if any(lk in s["train"] for s in specs.values())]
    Q, part, rec_sha = [], {}, {}
    for lk in looks:
        recs = sorted((AW3.LOOKS / lk).glob("record*.json"))
        if not recs:
            raise SystemExit(f"look {lk} has no record")
        for r in recs:
            rs = json.loads(r.read_text(encoding="utf-8"))["script_sha256"]
            if rs != AW3.LOOK_SCORE_SHA:
                raise SystemExit(f"{r} was not written by the pinned look scorer")
            rec_sha[str(r.relative_to(AW3.LOOKS))] = rs
        _ids, Ql = A16.load(AW3.LOOKS / lk)
        part[lk] = list(range(len(Q), len(Q) + len(Ql)))
        Q.extend(Ql)
        log(f"look {lk}: {len(Ql)} rows, {time.time() - t0:.0f}s")
    checks, _vocab = AW.anchor_tables(Q)
    log(f"anchors attached to {len(Q)} rows: {checks}, {time.time() - t0:.0f}s")
    x1 = part["x1"]
    B_rows = x1[1::2]
    M = np.stack([q["metrics"] for q in Q])[:, :, A16.RI]
    Bsel, Xsel = np.asarray(B_rows), np.asarray(x1)
    tB, gB = M[Bsel, 0], M[Bsel, 3]
    tX, gX = M[Xsel, 0], M[Xsel, 3]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    tr_all = sorted({i for sp in specs.values() for lk in sp["train"] for i in part[lk]})
    zT_of = [None] * len(Q)
    for i in tr_all:
        zT_of[i] = teacher_of(Q[i], a.teacher)
    TX = np.asarray([A16.metrics_of(teacher_of(Q[i], a.teacher), Q[i]["gold"], Q[i]["gt"]) for i in x1])
    TB = TX[1::2]                                              # B_rows = x1[1::2]
    ty = np.asarray([str(q["type"]) for q in Q])
    types = sorted(set(ty[Bsel].tolist()))
    if len(types) > 12:
        types = []
    W = np.random.default_rng(20261002).poisson(1.0, (1000, len(B_rows))).astype(np.float64)
    WX = np.random.default_rng(20261003).poisson(1.0, (1000, len(x1))).astype(np.float64)
    gain_B, gain_X = gB.mean(0) - tB.mean(0), gX.mean(0) - tX.mean(0)

    def rho(m, t, gain):
        return [round(float((m[:, j].mean() - t[:, j].mean()) / gain[j]), 3) if abs(gain[j]) > 1e-9 else None for j in range(3)]

    res = {"look": "anchor_walk8", "dataset": a.dataset, "script_sha256": AW.sha(Path(__file__)),
           "pins": {**AW6.PINS, "anchor_walk6": AW7.AW6_SHA, "anchor_walk7": AW7_SHA, "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA,
                    "phrase_table": phi_sha, "look_score_records": rec_sha}, "flag_checks": checks,
           "looks": {lk: len(r) for lk, r in part.items()}, "epochs": a.epochs, "lr": a.lr, "wd": a.wd,
           "kd": {"ce": a.ce, "kd": a.kd, "T": a.T, "teacher": a.teacher, "gnn_cols": list(GNN_COLS)},
           "B twin0 / gnn0": {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist()},
           "x1 twin0 / gnn0": {"twin0": tX.mean(0).round(4).tolist(), "gnn0": gX.mean(0).round(4).tolist()},
           "teacher_B": {"fit": TB.mean(0).round(4).tolist(), "rho_B (R@5, FC@5, hit@1)": rho(TB, tB, gain_B),
                         "minus_gnn0_B": AW3.boot_pair(TB - gB, W)},
           "teacher_x1": {"fit": TX.mean(0).round(4).tolist(), "rho_x1 (R@5, FC@5, hit@1)": rho(TX, tX, gain_X)}, "variants": {}}
    for lk in looks[1:]:
        S = np.asarray(part[lk])
        res[f"{lk} twin0 / gnn0"] = {"twin0": M[S, 0].mean(0).round(4).tolist(), "gnn0": M[S, 3].mean(0).round(4).tolist()}
    log(json.dumps({k: v for k, v in res.items() if "twin0" in k or k.startswith("teacher")}))

    def record(mB, mX, extra):
        return {**extra, "fit_B": mB.mean(0).round(4).tolist(), "rho_B (R@5, FC@5, hit@1)": rho(mB, tB, gain_B),
                "minus_gnn0_B": AW3.boot_pair(mB - gB, W), "minus_twin0_B": AW3.boot_pair(mB - tB, W), "fit_x1": mX.mean(0).round(4).tolist(),
                "rho_x1 (R@5, FC@5, hit@1)": rho(mX, tX, gain_X), "minus_gnn0_x1": AW3.boot_pair(mX - gX, WX),
                "by_type_B": {t: {"rows": int((ty[Bsel] == t).sum()), "fit": mB[ty[Bsel] == t].mean(0).round(4).tolist(),
                                  "twin0": tB[ty[Bsel] == t].mean(0).round(4).tolist(), "gnn0": gB[ty[Bsel] == t].mean(0).round(4).tolist()}
                              for t in types}}

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **{k.replace("@", "_s").replace("~", "__").replace(":", "_").replace("+", "-").replace("/", "."): x
                                                  for k, x in per_row.items()}, B_rows=np.asarray(B_rows))

    per_row, cache = {}, {}
    for name, sp in specs.items():
        t1 = time.time()
        key = (sp["tok"], sp["self"], sp["max_len"], sp["K"])
        if key not in cache:
            cache.clear()
            if sp["tok"] == "T0":
                nt, tk = 5, [A16.famdir(q) for q in Q]
            else:
                nt, tk = AW.anchor_tokens(Q, sp["K"], "w2")
            if sp["self"] is None:
                ntp = nt
                TY = [A16.walk_types(q, t, nt, sp["max_len"]) for q, t in zip(Q, tk)]
            else:
                ntp = 2 * nt + 1
                TY = []
                for q, t in zip(Q, tk):
                    ty_q = A16.walk_types(q, t, ntp, sp["max_len"])
                    sty = AW3.self_types(q, t, nt, ntp, sp["self"])
                    if set(ty_q) & set(sty):
                        raise SystemExit("a self type's code collides with a walk's")
                    ty_q.update(sty)
                    TY.append(ty_q)
            cache[key] = (ntp, TY)
        ntp, TY = cache[key]
        tr = [i for lk in sp["train"] for i in part[lk]]
        make = AW7.make_for(sp["model"], ntp, sp["K"], phi)
        models, seeds_out = [], {}
        for sd in sp["seeds"]:
            t2 = time.time()
            model, reads, best_ep, curve, margin, by_margin = fit_read_kd(Q, TY, ntp, tr, part["select"], {"B": B_rows, "x1": x1}, z_of, zT_of,
                                                                          make, a.epochs, sd, sp["rule"], a.lr, a.wd, a.ce, a.kd, a.T)
            models.append(model)
            nm = name if len(sp["seeds"]) == 1 else f"{name}#{sd}"
            per_row[nm], per_row[nm + "~x1"] = reads["B"], reads["x1"]
            seeds_out[str(sd)] = record(reads["B"], reads["x1"], {
                "best_epoch": best_ep, "curve": curve, "margin": None if margin is None else str(margin), "by_margin_select": by_margin,
                "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
                "gamma": float(model.gamma.item()) if hasattr(model, "gamma") else None,
                "gam": model.gam.detach().round(decimals=4).tolist() if hasattr(model, "gam") else None, "seconds": round(time.time() - t2, 1)})
            log(f"{nm}: {json.dumps({k: x for k, x in seeds_out[str(sd)].items() if k not in ('by_type_B', 'curve')})}")
        v = {**sp, "train_rows": len(tr), "tokens": ntp, "types_per_row": round(float(np.mean([len(TY[i]) for i in tr])), 1),
             "params": int(sum(p.numel() for p in models[0].parameters())), "seeds_read": seeds_out}
        if len(models) > 1:
            if sp["rule"] == "p":
                mg, by_margin = 0.0, None
            elif sp["rule"] == "np":
                mg, by_margin = None, None
            else:
                by_margin, best_v, mg = {}, -1.0, None
                for m_ in AW6.MARGINS:
                    val = float(AW7.read_ens(models, Q, TY, ntp, list(part["select"]), z_of, m_).mean())
                    by_margin[str(m_)] = round(val, 4)
                    if val > best_v + 1e-12:
                        mg, best_v = m_, val
            mB = AW7.read_ens(models, Q, TY, ntp, list(B_rows), z_of, mg)
            mX = AW7.read_ens(models, Q, TY, ntp, list(x1), z_of, mg)
            per_row[name], per_row[name + "~x1"] = mB, mX
            v["ensemble"] = record(mB, mX, {"margin": None if mg is None else str(mg), "by_margin_select": by_margin})
            log(f"{name} ensemble: {json.dumps({k: x for k, x in v['ensemble'].items() if k != 'by_type_B'})}")
        v["seconds"] = round(time.time() - t1, 1)
        res["variants"][name] = v
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


if __name__ == "__main__":
    main()
