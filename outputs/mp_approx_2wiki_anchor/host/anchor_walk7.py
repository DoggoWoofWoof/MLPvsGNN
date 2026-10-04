"""Design look (untracked; not a result and not filed): anchor-typed walks, round 7: relation vectors read from the anchor
phrase's text, seed ensembles, and the spread-scaled kappa.

Rounds 2 to 6 give each anchor token a free vector t1[k]: a learned relation, typed by the phrase but not read from it,
so a phrase seen rarely gets little signal and the vocabulary stops paying beyond the top 256 (round 2: A1024-1 and
A4096-1 read below A256-1). Here a structural token's vector is computed from its phrase: phi(k) is the phrase's
gte-Qwen2 vector (phrase_embed.py: document mode, the space of the served query embeddings; 0 for the 'other' rank),
mapped by a learned P (1536 -> 64):
    txt     t1[k] + P phi(k)                         the free table kept, the text added
    txt0    P phi(k) + D[famdir(k)]                  a phrase token from its text alone (and a vector per family x
                                                     direction); 'other', ner and knn keep a free vector
    txtc    txt0 plus gamma[famdir(k)] * <q, phi(k)> in the logit: the zero-shot query-relation match that KB GNNs
            read (cos(q, e_r)), gamma starting at 0
The logit <A q, e> is then a rank-64 bilinear match between the query and the relation's text, so a phrase borrows from
the phrases it reads like and K can grow: bases A{K}-1 for K in 256 and 1024 (one-edge walks; the phrases are the top K
of the dataset's w2 vocabulary). Two more levers on any model:
    sc      suffix: the bonus times the row's z spread^gamma (l16_look_gate2's 'scaled' kappa, gamma starting at 0)
    @0+1+2  a seed ensemble: each seed is fitted as a single variant is; the ensemble reads the mean of the seeds' final
            scores (under rule mp the margin is chosen again on the select rows for the mean)
Everything else (the fit loop, the rules, the selection, the reads) is anchor_walk6.py's, imported unchanged; with
--lr or --wd other than 3e-3 and 0 the same loops run with that learning rate and AdamW's decoupled decay on the matrices
and tables (parameters of two or more dimensions), the run's curves having peaked by epoch 1 to 3 at 3e-3.

    <train>:<base>[/<model>[/<rule>]][@seeds]    base T0, A256-1, A{K}-1 (K = 256 or 1024), A256; model as anchor_walk6's,
                                                 or txt, txt0, txtc, each optionally with 'sc'; linsc
    python outputs/mp_approx_2wiki_anchor/host/anchor_walk7.py --dataset 2wiki --variants x4+x5+x6:A1024-1/txt0 [--out PATH]
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import anchor_walk6 as AW6  # noqa: E402

AW5, AW4, AW3 = AW6.AW5, AW6.AW4, AW6.AW3
AW, G2, A16 = AW3.AW, AW3.G2, AW3.A16
AW6_SHA = "441d68bd85212d0bb1264b8ec269b9f5d440328b7869614c4287c67ca3ab49de"
PHRASE_K = 1024
PHRASE = {"2wiki": (ROOT / "outputs" / "mp_approx_2wiki_anchor" / "host" / "phrase_emb_w2_K1024.npy",
                    "0a965f0054ecfcecba457318256345ab2d1c3d9642a167a4ff2c63b3efce19d4"),
          "hotpotqa": (ROOT / "outputs" / "mp_approx_hotpot_anchor" / "host" / "phrase_emb_w2_K1024.npy",
                       "b2a69b2aa6be4b483d1b2fd509be6474f1dcc6d97269da4a78c1db725062c793"),
          "musique": (ROOT / "outputs" / "mp_approx_musique_anchor" / "host" / "phrase_emb_w2_K1024.npy",
                      "8ff27780a6911032f83860b5a7a3d6ce04800b493a09362ac706d4b143ef29d8")}
TXT = ("txt", "txt0", "txtc")
KS = (256, 1024)
log = A16.log


def famdir_of_tokens(nt, K):
    """Token -> family x direction (A16.famdir's codes): structural d * (K + 1) + rank -> d; ner 3; knn 4."""
    f = np.empty(nt, dtype=np.int64)
    f[:3 * (K + 1)] = np.repeat(np.arange(3), K + 1)
    f[3 * (K + 1)] = 3
    f[3 * (K + 1) + 1] = 4
    return f


class TxtMix(A16.Mix):
    """l16_look_analyze.Mix with a structural token's vector computed from its phrase's text vector (see the docstring)."""

    def __init__(self, nt, K, phi, kind, scaled=False, d=64, qdim=1536):
        super().__init__(nt, d, qdim)   # the same draws as Mix first
        if nt != 3 * (K + 1) + 2 or phi.shape[0] < K:
            raise SystemExit("TxtMix: the token count or the phrase table does not match K")
        Phi = np.zeros((nt, phi.shape[1]), dtype=np.float32)
        pm = np.zeros(nt, dtype=bool)
        for dd in range(3):
            Phi[dd * (K + 1):dd * (K + 1) + K] = phi[:K]
            pm[dd * (K + 1):dd * (K + 1) + K] = True
        self.register_buffer("Phi", torch.as_tensor(Phi))
        self.register_buffer("pmask", torch.as_tensor(pm))
        self.register_buffer("fdir", torch.as_tensor(famdir_of_tokens(nt, K)))
        self.kind = kind
        self.P = torch.nn.Linear(phi.shape[1], d, bias=False)
        if kind in ("txt0", "txtc"):
            self.D = torch.nn.Parameter(torch.zeros(5, d))
        if kind == "txtc":
            self.gam = torch.nn.Parameter(torch.zeros(5))
        if scaled:
            self.gamma = torch.nn.Parameter(torch.zeros(()))

    def tvec(self):
        base = self.P(self.Phi)
        if self.kind == "txt":
            return self.t1 + base
        return torch.where(self.pmask[:, None], base + self.D[self.fdir], self.t1)

    def forward(self, qemb, tb, t1, t2, tmask, ent_q, ent_t, ent_node, z, nmask):
        aq = self.A(qemb)
        L = (t2 > 0).long()
        e = self.bk[tb] + self.tvec()[t1] + self.t2[t2] + self.ln[L]
        w = (aq[:, None, :] * e).sum(-1) + self.c[tb, L]
        if self.kind == "txtc":
            cosv = qemb @ self.Phi.T                              # (B, nt); both sides unit (phrase vectors stored unit-L2)
            w = w + self.gam[self.fdir][t1] * cosv.gather(1, t1)
        w = w.masked_fill(~tmask, float("-inf"))
        null = aq @ self.nu + self.c_null
        return torch.softmax(torch.cat([w, null[:, None]], 1), 1)[:, :-1]


def make_for(model, ntp, K, phi):
    base, sc = (model[:-2], True) if model.endswith("sc") and model != "sc" else (model, False)
    if base in TXT:
        return lambda: TxtMix(ntp, K, phi, base, scaled=sc)
    if model == "linsc":
        return G2.make_for(None, True, ntp)
    return AW6.make_for(model, ntp)


def parse(name, train_looks):
    head, _at, seeds = name.partition("@")
    train, sep, rest = head.partition(":")
    if not sep:
        raise SystemExit(f"{name}: no train set")
    looks = train.split("+")
    if not looks or any(lk not in train_looks for lk in looks) or len(set(looks)) != len(looks):
        raise SystemExit(f"{name}: train looks must be among {train_looks}")
    parts = rest.split("/")
    if len(parts) > 3:
        raise SystemExit(f"{name}: too many fields")
    base, model, rule = parts[0], (parts[1] if len(parts) > 1 else "lin"), (parts[2] if len(parts) > 2 else "p")
    K = None
    if base in AW6.BASES:
        tok, selfk, max_len = AW6.BASES[base]
        K = AW3.K if tok == "A" else None
    elif base.startswith("A") and base.endswith("-1") and base[1:-2].isdigit() and int(base[1:-2]) in KS:
        tok, selfk, max_len, K = "A", None, 1, int(base[1:-2])
    else:
        raise SystemExit(f"{name}: unknown base {base}")
    mbase = model[:-2] if model.endswith("sc") and model not in ("sc", "linsc") else model
    if model != "linsc" and mbase not in TXT and model not in AW5.MODELS and model not in AW6.OPS:
        raise SystemExit(f"{name}: unknown model {model}")
    if mbase in TXT and (tok != "A" or max_len != 1 or selfk is not None):
        raise SystemExit(f"{name}: text models read one-edge anchor bases only")
    if K is not None and K != AW3.K and mbase not in TXT and model not in ("lin", "linsc"):
        raise SystemExit(f"{name}: K other than {AW3.K} only for lin and the text models")
    if rule not in AW6.RULES:
        raise SystemExit(f"{name}: unknown rule {rule}")
    sl = [int(s) for s in seeds.split("+")] if seeds else [0]
    if len(set(sl)) != len(sl):
        raise SystemExit(f"{name}: repeated seed")
    return {"train": looks, "base": base, "tok": tok, "self": selfk, "max_len": max_len, "K": K, "seeds": sl, "model": model, "rule": rule}


def fit_read_opt(Q, TY, nt, tr_rows, val_rows, read_sets, z_of, make, epochs, seed, rule, lr, wd, own=False):
    """anchor_walk6.fit_read_rule at lr 3e-3 and no decay (unchanged); otherwise its loops (rule p: anchor_walk3's, with
    the protect lines; np and mp: anchor_walk6's) with the optimiser below (own: this copy at any values, for the test)."""
    if not own and lr == 3e-3 and wd == 0.0:
        return AW6.fit_read_rule(Q, TY, nt, tr_rows, val_rows, read_sets, z_of, make, epochs, seed, rule)
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
            if rule == "p":
                first = G2.top1(PX[0][9])
                ar = torch.arange(len(rows))
                s = s.clone()
                s[ar, first] = float("-inf")
                gold = gold.clone()
                gold[ar, first] = 0.0
            keep = gold.sum(1) > 0
            if not bool(keep.any()):
                continue
            ls = torch.log_softmax(s[keep], 1)
            gk = gold[keep]
            g = gk / gk.sum(1, keepdim=True)
            loss = -(torch.where(gk > 0, ls, torch.zeros_like(ls)) * g).sum(1).mean()
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


def ens_scores(models, PX):
    s = None
    for m in models:
        si, gold = G2.scores(m, PX)
        s = si if s is None else s + si
    return s / len(models), gold


def read_ens(models, Q, TY, nt, rows, z_of, margin):
    """anchor_walk6.read_rule on the mean of the models' final scores."""
    for m in models:
        m.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = rows[s0:s0 + 256]
            PX = G2.LG.pack_gate(rr, Q, TY, z_of, nt)
            s, _gold = ens_scores(models, PX)
            if margin is not None:
                z = PX[0][9]
                ar = torch.arange(len(rr))
                first = G2.top1(z)
                zz = z.clone()
                zz[ar, first] = float("-inf")
                prot = (z[ar, first] - zz.max(1).values) >= margin
                s[ar[prot], first[prot]] = float("inf")
            for bi, i in enumerate(rr):
                out.append(A16.metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--variants", default="x4:A256-1/txt0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=AW3.EPOCHS)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    a = ap.parse_args(argv)
    if AW.sha(Path(AW6.__file__)) != AW6_SHA:
        raise SystemExit("anchor_walk6.py is not the pinned file")
    for mod, want in ((AW3, AW6.PINS["anchor_walk3"]), (AW4, AW6.PINS["anchor_walk4"]), (AW5, AW6.PINS["anchor_walk5"])):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    for mod, want in ((AW, AW3.AW_SHA), (G2, AW3.G2_SHA), (G2.LG, AW3.G1_SHA), (A16, AW.A16_SHA)):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    AW6.rebind(a.dataset)
    out_path = Path(a.out) if a.out else HERE / f"anchor_walk7_{a.dataset}.json"
    torch.set_num_threads(2)
    t0 = time.time()
    train_looks = AW6.TRAIN[a.dataset]
    specs = {name: parse(name, train_looks) for name in a.variants.split(",")}
    phi, phi_sha = None, None
    if any(s["model"].removesuffix("sc") in TXT for s in specs.values()):
        pth, want = PHRASE[a.dataset]
        phi_sha = AW.sha(pth)
        if want is None or phi_sha != want:
            raise SystemExit(f"{pth} is not the pinned phrase table")
        phi = np.load(pth).astype(np.float32)
        if phi.shape != (PHRASE_K, 1536):
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
    ty = np.asarray([str(q["type"]) for q in Q])
    types = sorted(set(ty[Bsel].tolist()))
    if len(types) > 12:
        types = []
    W = np.random.default_rng(20261002).poisson(1.0, (1000, len(B_rows))).astype(np.float64)
    WX = np.random.default_rng(20261003).poisson(1.0, (1000, len(x1))).astype(np.float64)
    res = {"look": "anchor_walk7", "dataset": a.dataset, "script_sha256": AW.sha(Path(__file__)),
           "pins": {**AW6.PINS, "anchor_walk6": AW6_SHA, "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA, "phrase_table": phi_sha,
                    "look_score_records": rec_sha}, "flag_checks": checks, "looks": {lk: len(r) for lk, r in part.items()}, "epochs": a.epochs,
           "lr": a.lr, "wd": a.wd,
           "B twin0 / gnn0": {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist()},
           "x1 twin0 / gnn0": {"twin0": tX.mean(0).round(4).tolist(), "gnn0": gX.mean(0).round(4).tolist()}, "variants": {}}
    for lk in looks[1:]:
        S = np.asarray(part[lk])
        res[f"{lk} twin0 / gnn0"] = {"twin0": M[S, 0].mean(0).round(4).tolist(), "gnn0": M[S, 3].mean(0).round(4).tolist()}
    log(json.dumps({k: v for k, v in res.items() if "twin0" in k}))
    gain_B, gain_X = gB.mean(0) - tB.mean(0), gX.mean(0) - tX.mean(0)

    def record(mB, mX, extra):
        return {**extra, "fit_B": mB.mean(0).round(4).tolist(),
                "rho_B (R@5, FC@5, hit@1)": [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain_B[j]), 3) if abs(gain_B[j]) > 1e-9 else None
                                             for j in range(3)],
                "minus_gnn0_B": AW3.boot_pair(mB - gB, W), "minus_twin0_B": AW3.boot_pair(mB - tB, W), "fit_x1": mX.mean(0).round(4).tolist(),
                "rho_x1 (R@5, FC@5, hit@1)": [round(float((mX[:, j].mean() - tX[:, j].mean()) / gain_X[j]), 3) if abs(gain_X[j]) > 1e-9 else None
                                              for j in range(3)],
                "minus_gnn0_x1": AW3.boot_pair(mX - gX, WX),
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
        make = make_for(sp["model"], ntp, sp["K"], phi)
        models, seeds_out = [], {}
        for sd in sp["seeds"]:
            t2 = time.time()
            model, reads, best_ep, curve, margin, by_margin = fit_read_opt(Q, TY, ntp, tr, part["select"], {"B": B_rows, "x1": x1}, z_of,
                                                                           make, a.epochs, sd, sp["rule"], a.lr, a.wd)
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
                    val = float(read_ens(models, Q, TY, ntp, list(part["select"]), z_of, m_).mean())
                    by_margin[str(m_)] = round(val, 4)
                    if val > best_v + 1e-12:
                        mg, best_v = m_, val
            mB = read_ens(models, Q, TY, ntp, list(B_rows), z_of, mg)
            mX = read_ens(models, Q, TY, ntp, list(x1), z_of, mg)
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
