"""Design look (untracked; not a result and not filed): lean_mlp3's fixed-set refits with weight averaging and seed
ensembles, against the overfitting the l3-2w refits show (select peaks at epoch 2 to 4 of 10 while the training loss keeps
falling). The carves, store, blocks, model, loss, batches, metrics and reads are lean_mlp3's and lean_mlp's, imported and
called unchanged; only the training loop is new, and with a 'base' config it is lean_mlp.fit's loop step for step.

Per fixed set, config and seed, one fit keeps three weight sets:
  best  the epoch with the best mean(R@5, FC@5) on select (lean_mlp.fit's choice; select-optimistic by construction),
  swa   the mean of the weights at the end of epochs swa_from .. epochs-1 (the choice reads no select number),
  last  the weights after the last epoch (select quality logged only).
A config is name=lr:wd:dropout:epochs:swa_from[:cos][:adamw][:drop]: Adam with L2 weight decay wd (lean_mlp.fit's form),
or AdamW's decoupled decay with ':adamw'; the model's dropout; a cosine-annealed lr (per batch, to 0 at the end) with
':cos'; with ':drop', lean_mlp.fit's block dropout over the set's blocks (rank always kept; ln-2w's dropout model read
+0.39 / +0.89 / +0.88 points over twin0 on 2wiki x1 where its fixed refit read +0.20 / +0.64 / -3.09). Every read keeps
all of the set's blocks.
An ensemble averages its members' scores over the seeds of one set and config (each member is base_w * z(rrf) + mlp, so
their scales agree). It needs the set's compiled features once and one forward per member, so its compile cost is the
set's; read_costs_ms carries the compile estimate only (lean_time2 times forwards).
Every read carries minus_ref, the paired difference against the reference member (the first set's first config's first
seed at its best epoch, which is lean_mlp.fit's refit), with its bootstrap interval.

Members are saved as plain lean_mlp3 entries (<set>/<config>/s<seed>@best and @swa), so lean_mlp3 --load-models and
lean_score.py read them unchanged; "ensembles" lists each ensemble's members.

    python outputs/mp_unified/lean_host.py lean_mlp5 --store pca256 --train 2wiki=fit --select 2wiki=select \
        --read 2wiki=x1 --sets lean2s --configs base=2e-3:1e-4:0.1:6:1;reg=1e-3:1e-2:0.2:8:2:cos:adamw --seeds 0,1,2 \
        --threads 2 --save-models outputs/mp_unified/lean/l5-2w_models.pt --out outputs/mp_unified/lean/l5-2w.json
    python outputs/mp_unified/lean_host.py lean_mlp5 --load-models outputs/mp_unified/lean/l5-2w_models.pt \
        --read hotpotqa=x1,musique=x1 --threads 2 --out outputs/mp_unified/lean/l5-2w-zs.json
    python outputs/mp_unified/lean_mlp5.py --selftest
"""
import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lean_mlp as LM  # noqa: E402
import lean_mlp2 as L2  # noqa: E402
import lean_mlp3 as L3  # noqa: E402

log = LM.log


# ── configs ──────────────────────────────────────────────────────────────────


def parse_configs(spec):
    """name=lr:wd:dropout:epochs:swa_from[:cos][:adamw][:drop];... -> {name: cfg}, in the order given."""
    out = {}
    for part in [p for p in spec.split(";") if p]:
        name, rest = part.split("=")
        tok = rest.split(":")
        flags = set(tok[5:])
        unknown = flags - {"cos", "adamw", "drop"}
        if len(tok) < 5 or unknown:
            raise SystemExit(f"config {part!r}: want lr:wd:dropout:epochs:swa_from[:cos][:adamw][:drop] (unknown {sorted(unknown)})")
        cfg = {"lr": float(tok[0]), "wd": float(tok[1]), "dropout": float(tok[2]), "epochs": int(tok[3]),
               "swa_from": int(tok[4]), "cos": "cos" in flags, "adamw": "adamw" in flags, "drop": "drop" in flags}
        if not 0 <= cfg["swa_from"] < cfg["epochs"]:
            raise SystemExit(f"config {part!r}: swa_from must be in [0, epochs)")
        out[name] = cfg
    return out


def cos_lr(lr, ep, k, n_units, epochs):
    """The cosine schedule at the batch that starts at unit k of epoch ep: lr at the start, 0 after the last batch."""
    t = (ep + k / max(n_units, 1)) / epochs
    return 0.5 * lr * (1.0 + math.cos(math.pi * t))


# ── the fit ──────────────────────────────────────────────────────────────────


def fit5(train, select, blocks, cfg, seed, hidden, keep_epochs=False):
    """lean_mlp.fit with the config's optimiser, dropout and schedule; returns the best-epoch, averaged and last weights,
    the curve and (keep_epochs) every epoch's weights. With dropout 0.1, wd 1e-4 and neither cos nor adamw the best
    weights are lean_mlp.fit's at the same lr (mode drop with ':drop', else fixed): the same seeding, model,
    permutations, keep draws, batches and steps, in the same order."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: train[0].widths[b] for b in blocks}
    model = LM.LeanMLP(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed)
    Opt = torch.optim.AdamW if cfg["adamw"] else torch.optim.Adam
    opt = Opt(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    units = [(ci, q) for ci, c in enumerate(train) for q in range(c.rows)]
    keep_all = None
    never = [blocks.index(b) for b in LM.NEVER if b in blocks]
    best, best_state, best_ep, curve, epochs = -1.0, None, -1, [], []
    acc, n_acc = None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(units))
        t0 = time.time()
        tot, nb = 0.0, 0
        for k in range(0, len(order), 32):
            if cfg["cos"]:
                for g in opt.param_groups:
                    g["lr"] = cos_lr(cfg["lr"], ep, k, len(order), cfg["epochs"])
            chunk = [units[j] for j in order[k:k + 32]]
            for ci in sorted({c for c, _ in chunk}):
                qs = np.asarray([q for c, q in chunk if c == ci])
                feats, nq, base_z, gold, _ = LM.batch_of(train[ci], qs, blocks)
                B = qs.size
                if cfg["drop"]:
                    r = rng.uniform(0.15, 1.0, size=(B, 1))
                    keep = (rng.uniform(size=(B, len(blocks))) < r).astype(np.float32)
                    keep[:, never] = 1.0
                    keep_t = torch.from_numpy(keep)
                else:
                    if keep_all is None or keep_all.shape[0] != B:
                        keep_all = torch.ones((B, len(blocks)), dtype=torch.float32)
                    keep_t = keep_all
                s = model(feats, keep_t, nq, B, base_z)
                loss = LM.listwise(s, gold, nq, B)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
        q = LM.quality(model, select, blocks, {b: 1.0 for b in blocks})
        state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        curve.append({"epoch": ep, "loss": tot / max(nb, 1), "select": q, "seconds": time.time() - t0,
                      "lr_end": opt.param_groups[0]["lr"]})
        log(f"  ep {ep}: loss {tot / max(nb, 1):.4f} select R@5/FC@5/hit@1 {[round(v, 4) for v in q]} ({time.time() - t0:.0f}s)")
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
        if keep_epochs:
            epochs.append(state)
    swa_state = {k_: (v / n_acc).to(torch.float32) for k_, v in acc.items()}
    last_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    return {"best": best_state, "swa": swa_state, "last": last_state, "best_epoch": best_ep, "curve": curve,
            "swa_epochs": list(range(cfg["swa_from"], cfg["epochs"])), "epoch_states": epochs, "widths": widths}


def build(blocks, widths, hidden, state):
    m = LM.LeanMLP(blocks, widths, hidden)
    m.load_state_dict(state)
    m.eval()
    return m


class Ens(nn.Module):
    """The mean of its members' scores; the members share one block set."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, feats, keep, nq, B, base_z):
        return torch.stack([m(feats, keep, nq, B, base_z) for m in self.members]).mean(0)


# ── reads ────────────────────────────────────────────────────────────────────


def read_models(a, models, ensembles, ref_name, out, carve):
    """Each member and ensemble on each --read carve, with minus_ref against the reference member's rows."""
    rng = np.random.default_rng(20261003)
    reads, read_costs, read_lean_ms, read_rows = {}, {}, {}, {}
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        c = carve(ds, cv)
        read_rows[key] = {"rows": c.rows, "chunks": c.chunks_read, "n_chunks": c.n_chunks, "carve_queries": c.carve_queries}
        (prof, share, lm), = L2.costs_of([c])
        read_lean_ms[key] = c.lean_ms
        sets = {}
        for name, (_m, bl, _k) in models.items():
            sets[name.split("/")[0]] = bl
        read_costs[key] = {s: LM.cost_ms(bl, prof, share, lm) for s, bl in sets.items()}
        log(f"read {key} ({c.rows} rows); approximate compile ms {({k: round(v, 2) for k, v in read_costs[key].items()})}")
        m0, bl0, keep0 = models[ref_name]
        r0, ref_rows = LM.read_one(m0, c, bl0, keep0, rng)
        items = [(n, models[n]) for n in models] + [(n, (Ens([models[x][0] for x in mem]),) + models[mem[0]][1:])
                                                     for n, mem in ensembles.items()]
        for name, (m, bl, keep) in items:
            if name == ref_name:
                r = r0
            else:
                r, _rows = LM.read_one(m, c, bl, keep, rng, ref_rows)
                r["minus_ref"] = r.pop("minus_full")
            reads[f"{name}@{key}"] = r
            extra = ("" if "minus_ref" not in r else " vs ref " + " ".join(
                f"{100 * p[0]:+.2f}[{100 * p[1][0]:+.2f},{100 * p[1][1]:+.2f}]" for p in r["minus_ref"]))
            log(f"  {name}@{key}: {LM.fmt(r)}{extra}")
        del c
    out["ref"] = ref_name
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["reads"] = reads


def ensembles_of(names):
    """<set>/<config>/ens@best and @swa over the seeds of each set and config (two or more seeds)."""
    groups = {}
    for n in names:
        head, kind = n.split("@")
        s, cfg, _seed = head.split("/")
        groups.setdefault((s, cfg, kind), []).append(n)
    return {f"{s}/{cfg}/ens@{kind}": mem for (s, cfg, kind), mem in groups.items() if len(mem) > 1}


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="pca256", choices=("rand128", "pca256"))
    ap.add_argument("--basis-from", default="")
    ap.add_argument("--fit-nodes", type=int, default=60000)
    ap.add_argument("--train", default="2wiki=fit")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--sets", default="lean2s", help="lean_mlp2's fixed sets (full, lean, lean2, lean2s)")
    ap.add_argument("--fixed", default="", help="more sets: name=blockA+blockB;name2=...")
    ap.add_argument("--configs", default="base=2e-3:1e-4:0.1:6:1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--save-models")
    ap.add_argument("--load-models", help="read only: the members and ensembles of this .pt file, on the --read carves")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    LM.LeanMLP = L3.LeanMLP3
    blob = torch.load(a.load_models, weights_only=False) if a.load_models else None
    store_name = blob["store"] if blob is not None else a.store
    L3.LeanMLP3.dim = L3.STORE_DIM if store_name == "pca256" else LM.PROJ_DIM
    names = sorted({ds for spec in (a.read,) + (() if blob is not None else (a.train, a.select)) for ds, _cv in LM.parse_sets(spec)})
    store, nodes, freeze, basis_info = None, {}, None, None
    if store_name == "pca256":
        if blob is not None:
            basis = blob["basis"]
            nodes, freeze = L3.open_nodes(names)
        else:
            src = a.basis_from or LM.parse_sets(a.train)[0][0]
            nodes, freeze = L3.open_nodes(sorted(set(names) | {src}))
            basis = L3.fit_basis(nodes[src], a.fit_nodes)
            basis["from"] = src
            log(f"basis from {src}: {basis['rows']} nodes, {basis['explained']:.3f} of the variance on {L3.STORE_K} axes "
                f"({basis['seconds']:.0f}s)")
        store = L3.Store(basis)
        basis_info = {k: v for k, v in basis.items() if k not in ("m", "V", "w")}

    def carve(ds, cv):
        return L3.Carve3(ds, cv, store, nodes.get(ds), a.limit)

    shas = {f"{n}_sha256": hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
            for n in ("lean_mlp", "lean_mlp2", "lean_mlp3")}
    shas["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if blob is not None:
        models = {name: (build(d["blocks"], d["widths"], d["hidden"], d["state"]), d["blocks"], d["keep"])
                  for name, d in blob["models"].items()}
        out = {"look": "lean_mlp5", "mode": "read_only", "store": blob["store"], "basis": basis_info, "freeze": freeze,
               "args": vars(a), "loaded": a.load_models,
               "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(), "fit_args": blob.get("args"),
               "configs": blob.get("configs"), "sets": blob.get("sets"), **shas}
        log(f"loaded {len(models)} members and {len(blob.get('ensembles', {}))} ensembles from {a.load_models}")
        read_models(a, models, blob.get("ensembles", {}), blob["ref"], out, carve)
        out["seconds"] = time.time() - t0
        L2.write(a, out)
        log(f"done in {time.time() - t0:.1f}s")
        return 0
    cfgs = parse_configs(a.configs)
    seeds = [int(s) for s in a.seeds.split(",") if s]
    tr = [carve(ds, cv) for ds, cv in LM.parse_sets(a.train)]
    se = [carve(ds, cv) for ds, cv in LM.parse_sets(a.select)]
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} ({time.time() - t0:.0f}s)")
    present = list(LM.COMPILED) + list(LM.LEAN) + list(L2.NEW)
    lean = [b for b in present if b in LM.NEVER or b in LM.LEAN]
    every = {"full": present, "lean": lean, "lean2": lean + [b for b in present if b in L2.NEW],
             "lean2s": lean + [b for b in present if b in L2.NEW and b not in L2.FULL_NEW]}
    sets = {k: list(v) for k, v in every.items() if k in a.sets.split(",")}
    for part in [p for p in a.fixed.split(";") if p]:
        name, bl = part.split("=")
        unknown = [b for b in bl.split("+") if b not in present]
        if unknown:
            raise SystemExit(f"set {name}: unknown blocks {unknown}")
        sets[name] = bl.split("+")
    used = sorted({b for bl in sets.values() for b in bl if b not in LM.LEAN})
    dead = [b for b in used if float(np.nanstd(np.concatenate([c.block(b, np.arange(int(c.off[-1]))).astype(np.float32)
                                                                for c in tr]))) == 0.0]
    for s in sets:
        sets[s] = [b for b in sets[s] if b not in dead]
    log(f"sets {sets}; constant on the training rows (dropped) {dead}")
    costs = L2.costs_of(tr)
    out = {"look": "lean_mlp5", "store": store_name, "basis": basis_info, "freeze": freeze, "args": vars(a), "configs": cfgs,
           "seeds": seeds, "sets": sets, "dead": dead, "set_costs_ms": {s: LM.cost_multi(bl, costs) for s, bl in sets.items()},
           "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se},
           "profiles": {c.ds: LM.profile(c.ds) for c in tr}, "struct_share": {f"{c.ds}={c.carve}": c.struct_share for c in tr},
           "assumptions": ["lean_mlp3's cost assumptions", "an ensemble's compile cost is its set's; its forward is one per member"],
           "fits": {}, **shas}
    models = {}
    for s, bl in sets.items():
        for cname, cfg in cfgs.items():
            for seed in seeds:
                tag = f"{s}/{cname}/s{seed}"
                log(f"fit {tag} ({len(bl)} blocks, {out['set_costs_ms'][s]:.2f} ms; {cfg})")
                t = time.time()
                f = fit5(tr, se, bl, cfg, seed, a.hidden)
                rec = {"best_epoch": f["best_epoch"], "swa_epochs": f["swa_epochs"], "curve": f["curve"], "seconds": time.time() - t}
                for kind in ("best", "swa", "last"):
                    m = build(bl, f["widths"], a.hidden, f[kind])
                    rec[f"select_{kind}"] = LM.quality(m, se, bl, {b: 1.0 for b in bl})
                    if kind != "last":
                        models[f"{tag}@{kind}"] = (m, bl, {b: 1.0 for b in bl})
                log(f"  {tag}: select best (ep {f['best_epoch']}) {[round(v, 4) for v in rec['select_best']]}, swa "
                    f"{[round(v, 4) for v in rec['select_swa']]}, last {[round(v, 4) for v in rec['select_last']]}")
                out["fits"][tag] = rec
    ensembles = ensembles_of(list(models))
    for name, mem in ensembles.items():
        m = Ens([models[x][0] for x in mem])
        bl = models[mem[0]][1]
        q = LM.quality(m, se, bl, {b: 1.0 for b in bl})
        out.setdefault("select_ensembles", {})[name] = q
        log(f"  {name}: select {[round(v, 4) for v in q]}")
    ref_name = f"{next(iter(sets))}/{next(iter(cfgs))}/s{seeds[0]}@best"
    if a.save_models:
        Path(a.save_models).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"models": {name: {"state": m.state_dict(), "blocks": bl, "keep": keep, "widths": m.widths, "hidden": a.hidden}
                               for name, (m, bl, keep) in models.items()},
                    "ensembles": ensembles, "ref": ref_name, "pick": None, "present": present, "dead": dead, "args": vars(a),
                    "configs": cfgs, "sets": sets, "store": store_name, "basis": None if store is None else store.basis},
                   a.save_models)
        log(f"saved {len(models)} members and {len(ensembles)} ensembles to {a.save_models}")
    del tr, se
    read_models(a, models, ensembles, ref_name, out, carve)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


class ToyCarve:
    """The attributes lean_mlp.batch_of, quality and read_one read, on random pools (gold tied to two features)."""

    def __init__(self, rows, seed):
        rng = np.random.default_rng(seed)
        self.ds, self.carve, self.rows = "toy", f"t{seed}", rows
        self.n = rng.integers(6, 14, size=rows)
        self.off = np.concatenate([[0], np.cumsum(self.n)])
        N = int(self.off[-1])
        self.widths = {"rank": 3, "WALK": 4, "SEMB": LM.SEMB_DIM}
        self._b = {"rank": rng.standard_normal((N, 3)).astype(np.float32), "WALK": rng.standard_normal((N, 4)).astype(np.float32)}
        self.proj = rng.standard_normal((N, LM.PROJ_DIM)).astype(np.float16)
        self.q_emb = rng.standard_normal((rows, 1536)).astype(np.float16)
        self.x = rng.standard_normal((N, 2)).astype(np.float32)
        self.rrf = 0
        sig = self._b["rank"][:, 0] + self._b["WALK"][:, 1] + 0.3 * rng.standard_normal(N)
        self.gold = np.zeros(N, np.int64)
        for i in range(rows):
            a, b = self.off[i], self.off[i + 1]
            self.gold[a + np.argsort(-sig[a:b])[:2]] = 1
        self.gold_total = np.full(rows, 2)
        self.score = rng.standard_normal((N, 4)).astype(np.float32)

    def block(self, b, idx):
        return self._b[b][idx]


def selftest():
    torch.set_num_threads(1)
    tr, se = [ToyCarve(70, 1)], [ToyCarve(30, 2)]
    blocks = ["rank", "WALK", "SEMB"]
    # with the base config the best weights are lean_mlp.fit's, bit for bit
    base = parse_configs("base=3e-2:1e-4:0.1:4:1")["base"]
    ref, _curve = LM.fit(tr, se, blocks, "fixed", 4, 3e-2, 0, 32)
    f = fit5(tr, se, blocks, base, 0, 32, keep_epochs=True)
    assert f["best_epoch"] > 0, "the toy must learn past epoch 0 for the comparison to bite"
    for k, v in ref.state_dict().items():
        assert torch.equal(v, f["best"][k]), f"best weights differ from lean_mlp.fit on {k}"
    # the averaged weights are the mean of epochs swa_from .. epochs-1; last is the final epoch's
    for k in f["swa"]:
        want = torch.stack([e[k].double() for e in f["epoch_states"][1:]]).mean(0).to(torch.float32)
        assert torch.allclose(f["swa"][k], want, atol=1e-7), k
        assert torch.equal(f["last"][k], f["epoch_states"][-1][k]), k
    assert f["swa_epochs"] == [1, 2, 3]
    # with ':drop' the best weights are lean_mlp.fit's in mode drop
    refd, _curve = LM.fit(tr, se, blocks, "drop", 4, 3e-2, 2, 32)
    fd = fit5(tr, se, blocks, parse_configs("d=3e-2:1e-4:0.1:4:1:drop")["d"], 2, 32)
    for k, v in refd.state_dict().items():
        assert torch.equal(v, fd["best"][k]), f"drop: best weights differ from lean_mlp.fit on {k}"
    # cosine: lr at the first batch, 0 after the last, half at mid-run; the curve records the last batch's lr
    assert cos_lr(1.0, 0, 0, 10, 4) == 1.0 and abs(cos_lr(1.0, 2, 0, 10, 4) - 0.5) < 1e-12 and abs(cos_lr(1.0, 4, 0, 10, 4)) < 1e-12
    fc = fit5(tr, se, blocks, parse_configs("c=2e-3:1e-2:0.2:3:0:cos:adamw")["c"], 1, 32)
    assert 0 < fc["curve"][-1]["lr_end"] < 2e-3 * 0.05, fc["curve"][-1]["lr_end"]
    # an ensemble's scores are the mean of its members'
    m1, m2 = build(blocks, f["widths"], 32, f["best"]), build(blocks, f["widths"], 32, fc["swa"])
    e = Ens([m1, m2])
    s1, s2, se_ = (LM.scores_of(m, se[0], blocks, {}) for m in (m1, m2, e))
    assert np.allclose(se_, 0.5 * (s1 + s2), atol=1e-6), "ensemble mean"
    # names: ensembles group by set and config over seeds; a lone seed makes none
    names = ["a/base/s0@best", "a/base/s1@best", "a/base/s0@swa", "a/base/s1@swa", "b/reg/s0@best"]
    assert ensembles_of(names) == {"a/base/ens@best": names[:2], "a/base/ens@swa": names[2:4]}
    # a saved member loads as lean_mlp3.read_only builds it
    L3.LeanMLP3.dim = LM.PROJ_DIM
    m3 = L3.LeanMLP3(blocks, f["widths"], 32)
    m3.load_state_dict(m1.state_dict())
    m3.eval()
    assert np.array_equal(LM.scores_of(m3, se[0], blocks, {}), s1)
    for bad in ("x=1e-3:0:0.1:4", "x=1e-3:0:0.1:4:4", "x=1e-3:0:0.1:4:1:sgd"):
        try:
            parse_configs(bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    print("selftest ok: base and drop configs = lean_mlp.fit (fixed, drop) bit for bit; swa = mean of the kept epochs; cosine end points; "
          "ensemble = mean of members; ensemble names; a member loads as lean_mlp3 builds it; bad configs refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
