"""Design look (untracked; not a result and not filed): one text-coded relation model fitted on several graphs at once,
passage graphs and knowledge graphs mixed, each graph keeping its own edge-label table; read on every graph it was
fitted on. anchor_few reads its models zero-shot (--n 0) and few-shot on the graphs it never saw.

A text, gen or cos model reads an edge label only through the label's text vector (an anchor phrase on a passage graph,
a relation name on a KB), so one model can be fitted on any mix of graphs: before each batch the model is shown that
batch's graph's table (anchor_gen3.set_phi) and nothing else changes. T0 (family x direction, no label) is the
label-free control. A passage graph reads anchor_gen3's 4096-phrase table (phrase ranks as its anchors give them), a
KB its relation table (anchor_gen_kb.kb_relations, ranked by their count over the graph's training rows' structural
edges, label-free, as anchor_kbfit ranks them).
Training: each graph's --train carves; a batch never mixes graphs. --balance graph (the default) gives every graph the
same number of batches per epoch, the smallest graph's count: the smallest graph passes its rows once per epoch and
every larger graph takes full batches of 128 from its own cycling permutation. --balance rows passes every row of every
graph once per epoch. With more than one graph the batch order is shuffled across graphs each epoch. The epoch is chosen
on the mean over graphs of each graph's select-carve score (rule p, scored as anchor_walk8.fit_read_kd scores it), so
every graph weighs the same. Loss, batches, AdamW groups, rule p and reads are fit_read_kd's. With one graph the loop is
fit_read_kd's, draw for draw: the smoke test checks it bit for bit against anchor_gen's, anchor_gen3's, anchor_cos's and
anchor_kbfit's fits.
Reads on every training graph's x1 half B, against its twin and GNN, each paired against ID:
    ID    the graph's own view
    NR    every edge 'other' (no relation information)
    SHUF  (a KB) every relation shown with the next relation's text
    SWAP  (two-edge) every walk (b, t1, t2) read as (b, t2, t1) (anchor_order.swap_types); exactly 0 for commutative ops
Models are saved as anchor_gen3 saves them (T0 as anchor_gen does), their dataset the '+'-joined training graphs, so
anchor_few reads them on any graph outside that set. anchor_few refuses only a model whose dataset IS its target, so a
joint model must be sent only to graphs outside its training set. PCA codes are refused (a basis is one graph's), as
are webqsp as a training graph (it stays the unseen KB) and the carves reserved for declared reads (2wiki x2 / x3,
musique x3). Nothing reads a neighbour's score or state. The out name must carry 'univ'.

    python outputs/mp_approx_2wiki_anchor/host/anchor_univ.py --train 2wiki=x4,hotpotqa=x4,metaqa=fit \
        --variants T0,A256-1/glinc,A256/gropet --out outputs/mp_approx_kb_anchor/host/anchor_univ_j3_x.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_kb_anchor" / "host"))
import anchor_cos as AC  # noqa: E402
import anchor_order as AO  # noqa: E402

G3 = AC.G3
AG, G2N = G3.AG, G3.G2N
AW11, AW8, AW6, A16, AW, AW3 = AG.AW11, AG.AW8, AG.AW6, AG.A16, AG.AW, AG.AW3
G2 = AW8.G2
R_TOP = AG.R_TOP
PINS = {"anchor_cos": "1039cf2f043bc6e3c5d342d0ea8f8f89ee3a1873c6e78a94f4078e99b7f667bf",
        "anchor_gen3": "ee5f76dc7d391649e04ab2b3575d4fdd30dd62cd2a34b1b7dbd114d56f7b14a7",
        "anchor_gen": "bdc938f546cff1192dda82701e92cf7003fce8aded07831c2c7930119ce4f108",
        "anchor_gen2": "cbb976408648e2da0d5398df7310b2695a471e246f7def05413f74cec84711c6",
        "anchor_order": "467d215b4d6c25f28ff912b2f08b97c67e28ccb5277d066a33e189ed243baea2"}
KB_PINS = {"anchor_kbfit": "c0f7eedc6d4563a4b43febe42874d26760f5563d93ac4d0544c966634bfc09f7",
           "anchor_gen_kb": "4e09d59de0a60767eb591863352389ff99e164f0e2cd841b858debd309ee7952"}
PASSAGE = ("2wiki", "hotpotqa", "musique")   # anchor_walk6.rebind('2wiki') changes nothing, so 2wiki loads first
KB_TRAIN = {"metaqa": ("fit",)}
TAG = "j"
log = AG.log


def check_pins():
    for mod, key in ((AC, "anchor_cos"), (G3, "anchor_gen3"), (AG, "anchor_gen"), (G2N, "anchor_gen2"), (AO, "anchor_order")):
        if AW.sha(Path(mod.__file__)) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    AG.check_pins()


def train_looks(ds):
    if ds in KB_TRAIN:
        return KB_TRAIN[ds]
    return AW6.TRAIN[ds] if ds != "2wiki" else ("fit", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12")


def parse_train(s):
    out = {}
    for part in s.split(","):
        ds, _eq, looks = part.partition("=")
        if ds == "webqsp":
            raise SystemExit("--train: webqsp stays the unseen KB")
        if ds not in PASSAGE and ds not in KB_TRAIN:
            raise SystemExit(f"--train {part}: a dataset among {PASSAGE + tuple(KB_TRAIN)}")
        lk = looks.split("+") if looks else []
        if ds in out or not lk or any(x not in train_looks(ds) for x in lk) or len(set(lk)) != len(lk):
            raise SystemExit(f"--train {part}: each dataset once, its looks among {train_looks(ds)}")
        out[ds] = lk
    return {ds: out[ds] for ds in [d for d in tuple(KB_TRAIN) + PASSAGE if d in out]}


def parse_variant(v):
    if ":" in v:
        raise SystemExit(f"{v}: a variant names no train looks (--train gives them)")
    name = f"{TAG}:{v}"
    base = v.split("/")[0]
    sp = AG.parse(name, (TAG,)) if base.startswith("T0") else G3.parse(name, (TAG,))
    if sp.get("pca"):
        raise SystemExit(f"{v}: a PCA basis is one graph's; refused")
    if sp["rule"] != "p":
        raise SystemExit(f"{v}: rule {sp['rule']}; this look selects as rule p does")
    if sp["family"] == "free" and sp["tok"] != "T0":
        raise SystemExit(f"{v}: a free per-phrase model has no vocabulary shared across graphs")
    return sp


def load_graphs(trains, max_len, t0, need_phi):
    """Every training graph's x1, select and training carves, KBs first (their loader reads no rebind state), then the
    passage graphs in PASSAGE order. One row list; part[(ds, carve)] its rows."""
    Q, part, G = [], {}, {}
    kbs = [ds for ds in trains if ds in KB_TRAIN]
    if kbs:
        import anchor_kbfit as KF
        KB = KF.KB
        for mod, key in ((KF, "anchor_kbfit"), (KB, "anchor_gen_kb")):
            if AW.sha(Path(mod.__file__)) != KB_PINS[key]:
                raise SystemExit(f"{mod.__file__} is not the pinned file")
    for ds in kbs:
        AW11.STATE.update({"max_len": max_len, "proj": False, "rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0})
        Q0, p0, rec_sha, rel = [], {}, {}, None
        for carve in ["x1", "select"] + trains[ds]:
            Ql, rs, rl = KF.load_carve(ds, carve)
            if rel is not None and rl != rel:
                raise SystemExit("the carves disagree on the relation bank")
            rel = rl
            p0[carve] = list(range(len(Q0), len(Q0) + len(Ql)))
            Q0.extend(Ql)
            rec_sha.update(rs)
            log(f"{ds}/{carve}: {len(Ql)} rows, {time.time() - t0:.0f}s")
        load_state = dict(AW11.STATE)
        tr0 = [i for lk in trains[ds] for i in p0[lk]]
        rank_of, vocab, phi, counts = KB.kb_relations(ds, Q0, tr0, int(rel["n_relations"]))
        Qv, st = KB.kb_view(Q0, rank_of)
        del Q0
        off = len(Q)
        Q.extend(Qv)
        for lk, rows in p0.items():
            part[(ds, lk)] = [off + i for i in rows]
        G[ds] = {"kind": "kb", "phi": phi, "phi_sha": KB.RELS[ds][1], "n_rel": len(vocab), "vocab": vocab,
                 "info": {"rel_vocab": KB.RELS[ds][0], "rel_embeddings": KB.RELS[ds][1], "look_score_kb": KB.LOOK_KB_SHA,
                          "look_score_kb_records": rec_sha, "relations": {**rel, "ranked_by_train_rows": counts[:64], "n_ranked": len(counts)},
                          "kb_view": st, "pruned_loader": load_state}}
    for ds in [d for d in PASSAGE if d in trains]:
        AW6.rebind(ds)
        Qd, pd, rec_sha, load_state = AG.load_looks(["x1", "select"] + trains[ds], max_len, t0)
        checks, voc = AW.anchor_tables(Qd)
        log(f"{ds}: anchors attached to {len(Qd)} rows: {checks}, {time.time() - t0:.0f}s")
        phi, phi_sha = G3.phrase_table(ds) if need_phi else (None, None)
        off = len(Q)
        Q.extend(Qd)
        for lk, rows in pd.items():
            part[(ds, lk)] = [off + i for i in rows]
        G[ds] = {"kind": "passage", "phi": phi, "phi_sha": phi_sha, "n_rel": None, "vocab": list(voc.get("w2", [])),
                 "info": {"phrase_table": phi_sha, "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA, "look_score_records": rec_sha,
                          "flag_checks": checks, "pruned_loader": load_state}}
        del Qd
    return Q, part, G


def set_graph(model, sp, phi):
    if sp["family"] != "free":
        G3.set_phi(model, phi, sp["K"])


def epoch_batches(rng, names, tr, balance, streams):
    """This epoch's (graph, rows) batches. One graph, or --balance rows: each graph's rows once, in fit_read_kd's
    draw (one permutation per graph per epoch). --balance graph: the smallest graph's rows once, every other graph the
    same number of full batches from its cycling permutation."""
    nb = {g: math.ceil(len(tr[g]) / 128) for g in names}
    out = []
    if balance == "rows" or len(names) == 1:
        for g in names:
            perm = rng.permutation(len(tr[g]))
            out += [(g, [tr[g][j] for j in perm[s0:s0 + 128]]) for s0 in range(0, len(tr[g]), 128)]
    else:
        g_min = min(names, key=lambda g: nb[g])
        S = nb[g_min]
        for g in names:
            if g == g_min:
                perm = rng.permutation(len(tr[g]))
                out += [(g, [tr[g][j] for j in perm[s0:s0 + 128]]) for s0 in range(0, len(tr[g]), 128)]
                continue
            st = streams[g]
            for _ in range(S):
                if st["perm"] is None or st["pos"] + 128 > len(tr[g]):
                    st["perm"], st["pos"] = rng.permutation(len(tr[g])), 0
                out.append((g, [tr[g][j] for j in st["perm"][st["pos"]:st["pos"] + 128]]))
                st["pos"] += 128
    if len(names) > 1:
        order = rng.permutation(len(out))
        out = [out[k] for k in order]
    return out


def fit_joint(Q, TY, nt, sp, G, names, tr, sel, z_of, make, epochs, seed, lr, wd, balance):
    """fit_read_kd's loop (ce 1, no teacher, rule p) over several graphs; per batch the model sees its graph's table."""
    torch.manual_seed(seed)
    model = make()
    mats = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    rng = np.random.default_rng(seed)
    streams = {g: {"perm": None, "pos": 0} for g in names}
    best, best_state, best_ep, curve, by_graph, n_batches = -1.0, None, -1, [], {g: [] for g in names}, []
    for ep in range(epochs):
        batches = epoch_batches(rng, names, tr, balance, streams)
        n_batches.append({g: sum(1 for b in batches if b[0] == g) for g in names})
        model.train()
        for g, rows in batches:
            set_graph(model, sp, G[g]["phi"])
            PX = G2.LG.pack_gate(rows, Q, TY, z_of, nt)
            s, gold = G2.scores(model, PX)
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
            gw = gk / gk.sum(1, keepdim=True)
            loss = 1.0 * -(torch.where(gk > 0, ls, torch.zeros_like(ls)) * gw).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sc = {}
        for g in names:
            set_graph(model, sp, G[g]["phi"])
            sc[g] = float(G2.read_rows(model, Q, TY, nt, list(sel[g]), z_of)[:, :2].mean())
            by_graph[g].append(round(sc[g], 4))
        score = sc[names[0]] if len(names) == 1 else float(np.mean([sc[g] for g in names]))
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"  ep {ep}: select {round(score, 4)} " + " ".join(f"{g} {sc[g]:.4f}" for g in names))
    model.load_state_dict(best_state)
    return model, best_ep, curve, by_graph, n_batches


def run(a):
    out_path = Path(a.out)
    if "univ" not in out_path.name:
        raise SystemExit("--out must carry 'univ' in its file name")
    if a.balance not in ("graph", "rows"):
        raise SystemExit("--balance graph|rows")
    trains = parse_train(a.train)
    names = list(trains)
    specs = {v: parse_variant(v) for v in a.variants.split(",")}
    torch.set_num_threads(2)
    t0 = time.time()
    max_len = max(sp["max_len"] for sp in specs.values())
    need_phi = any(sp["family"] != "free" for sp in specs.values())
    Q, part, G = load_graphs(trains, max_len, t0, need_phi)
    for g in names:
        if G[g]["kind"] == "passage":
            Ks = [sp["K"] for sp in specs.values() if sp["tok"] == "A"]
            if Ks and len(G[g]["vocab"]) < max(Ks):
                raise SystemExit(f"{g}: the compact's phrase strings do not cover K")
    tr = {g: [i for lk in trains[g] for i in part[(g, lk)]] for g in names}
    sel = {g: part[(g, "select")] for g in names}
    B = {g: part[(g, "x1")][1::2] for g in names}
    RB = {g: AG.Reader(Q, B[g], 20261002) for g in names}
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "anchor_univ", "train": trains, "script_sha256": AW.sha(Path(__file__)), "pins": PINS,
           "kb_pins": KB_PINS if any(G[g]["kind"] == "kb" for g in names) else None, "balance": a.balance, "epochs": a.epochs,
           "lr": a.lr, "wd": a.wd,
           "graphs": {g: {"kind": G[g]["kind"], "train_rows": len(tr[g]), "select_rows": len(sel[g]), "B_rows": len(B[g]),
                          "batches_per_epoch": math.ceil(len(tr[g]) / 128), "info": G[g]["info"],
                          "vocab_head": [str(x) for x in G[g]["vocab"][:12]], "B": RB[g].base()} for g in names},
           "variants": {}}
    per_row = {}
    model_dir = out_path.parent / (out_path.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset = "+".join(names)

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, **{f"B_rows_{g}": np.asarray(B[g]) for g in names})

    cache = {}
    for vi, (name, sp) in enumerate(specs.items()):
        t1 = time.time()
        K = sp["K"]
        key = (sp["tok"], K, sp["max_len"])
        if key not in cache:
            cache.clear()
            TY, nt = [None] * len(Q), None
            for g in names:
                rows_g = sorted({i for lk in ["x1", "select"] + trains[g] for i in part[(g, lk)]})
                nt, ty = AG.types_for(Q, sp, None if sp["tok"] == "T0" else AG.identity_map(K), rows_g)
                for i in rows_g:
                    TY[i] = ty[i]
                del ty
            cache[key] = (nt, TY)
        nt, TY = cache[key]
        alt = {g: {} for g in names}
        for g in names:
            if sp["tok"] == "A":
                alt[g]["NR"] = AG.types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B[g])[1]
            if sp["max_len"] == 2:
                alt[g]["SWAP"] = AO.swap_types(TY, nt, B[g])
        make = G3.make_for(sp, nt, G[names[0]]["phi"], None)
        seeds_out, m0 = {}, None
        for sd in sp["seeds"]:
            t2 = time.time()
            model, best_ep, curve, by_graph, n_batches = fit_joint(Q, TY, nt, sp, G, names, tr, sel, z_of, make, a.epochs, sd, a.lr, a.wd,
                                                                   a.balance)
            if m0 is None:
                m0 = model
            rd = {}
            for g in names:
                set_graph(model, sp, G[g]["phi"])
                R = RB[g]
                mID = AG.read(model, Q, TY, nt, B[g], z_of, 0.0)
                per_row[f"{vi}_{sd}_{g}_ID"] = mID
                e = {"ID": R.record(mID)}
                for nm, TYa in alt[g].items():
                    m_alt = AG.read(model, Q, TYa, nt, B[g], z_of, 0.0)
                    per_row[f"{vi}_{sd}_{g}_{nm}"] = m_alt
                    e[nm], e[f"{nm} - ID"] = R.record(m_alt, by_type=False), AW3.boot_pair(m_alt - mID, R.W)
                if G[g]["kind"] == "kb" and sp["tok"] == "A" and sp["family"] != "free":
                    phi = G[g]["phi"]
                    phi_s = phi.copy()
                    k_ = min(G[g]["n_rel"], R_TOP)
                    phi_s[:k_] = phi[(np.arange(k_) + 1) % k_]
                    set_graph(model, sp, phi_s)
                    m_alt = AG.read(model, Q, TY, nt, B[g], z_of, 0.0)
                    set_graph(model, sp, phi)
                    per_row[f"{vi}_{sd}_{g}_SHUF"] = m_alt
                    e["SHUF"], e["SHUF - ID"] = R.record(m_alt, by_type=False), AW3.boot_pair(m_alt - mID, R.W)
                rd[g] = e
            pt = model_dir / f"v{vi}_s{sd}.pt"
            saver = G3.save_model if "pca" in sp else AG.save_model
            saver(pt, model, sp, nt, 0.0, dataset, [], {g: G[g]["phi_sha"] for g in names},
                  {"hold": None, "name": name, "seed": sd, "rdrop": None,
                   "univ": {"train": trains, "balance": a.balance, "epochs": a.epochs, "lr": a.lr, "wd": a.wd}})
            seeds_out[str(sd)] = {"best_epoch": best_ep, "curve": curve, "curve_by_graph": by_graph, "batches_per_epoch": n_batches[0],
                                  "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
                                  "model_file": str(pt.relative_to(ROOT)) if pt.is_relative_to(ROOT) else str(pt), "reads": rd,
                                  "seconds": round(time.time() - t2, 1)}
            log(f"{name}#{sd} ({sp['family']}): ep {best_ep}, " + "; ".join(
                f"{g} " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in rd[g].items() if isinstance(v, dict) and "rho (R@5, FC@5, hit@1)" in v)
                for g in names))
            res["variants"][name] = {**sp, "tokens": nt, "params": int(sum(p.numel() for p in m0.parameters())),
                                     "params_used": AG.params_used(m0, sp), "seeds_read": seeds_out, "seconds": round(time.time() - t1, 1)}
            save()
        del alt
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--balance", default="graph")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    a = ap.parse_args(argv)
    check_pins()
    AC.bind()
    run(a)


if __name__ == "__main__":
    main()
