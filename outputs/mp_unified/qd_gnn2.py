"""Design look (untracked; not a result and not filed): qd_gnn.py's arms and packing, imported from the pinned file
and unchanged, in qd_gnn's shared loop (the same batches, loss and selection for rules np and p), with these additions.

(1) 2wiki reads look_x_six's six-pair looks (qd_six's rebinding; --looks-2w pilot keeps level 16's pilot-pair looks),
    so every graph's twin0 / gnn0 is the six-dataset pair.
(2) --read: graphs never trained or selected on, read zero-shot on anchor_few's read carves: x1's half B (webqsp, whose
    fit stride is 1 and so has no x1: its whole select carve). A read KB's relations are ranked by their count over the
    read carve's structural edges (label-free; anchor_few's convention); a read passage graph keeps its own phrase
    table. A label arm is read ID (its own labels, through their text) and NR (every label 'other': the label-free
    fallback); an arm whose labels are one graph's ranks (QD-ID) is read NR only there.
(3) Rule both: the loss is np's plus p's (the whole pool's listwise loss plus the loss with the twin's rank-1 node
    dropped from the softmax and the golds), and the epoch is selected as np selects (the select carve's mean of
    recall@5, full_coverage@5 and hit@1 read without protection).
(4) Every arm is also read ID/mp: the twin's rank-1 node placed first on the rows where its z margin over rank 2 is at
    least m, m chosen from anchor_walk6.MARGINS on the training graphs' select carves (their mean; 0 is rule p's read,
    inf rule np's).
(5) Every read carries abs = fit / gnn0 for (recall@5, full_coverage@5, hit@1): the absolute fraction of the six-pair
    GNN's metric, beside rho (the fraction of the twin-to-GNN gap).

    python outputs/mp_unified/qd_gnn2.py --train 2wiki=x4,hotpotqa=x4,metaqa=fit --read musique,webqsp \
        --arms QD-T0-L3,QD-TXT-L3,W:T0 --rule both --out outputs/mp_unified/qd/j3.json
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_six as Q6  # noqa: E402  (imports qd_gnn, which sets the BLAS thread counts before numpy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

QG = Q6.QG
log, sha = QG.log, QG.sha
READ = {"2wiki": "x1", "hotpotqa": "x1", "musique": "x1", "metaqa": "x1", "webqsp": "select"}
KBS = ("metaqa", "webqsp")
WHOLE = ("webqsp",)
MARGINS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, math.inf)   # anchor_walk6.MARGINS (checked at run time)
RULES = ("np", "p", "both")


def load_all(trains, reads, max_len, t0, AU):
    """anchor_univ.load_graphs for the training graphs (x1, select and their training carves; a KB's relations ranked
    over its training rows), plus each read graph's read carve alone. KBs first, then the passage graphs in
    anchor_univ.PASSAGE order (2wiki first, as anchor_walk6.rebind needs)."""
    AG, AW, AW6, AW11, G3 = AU.AG, AU.AW, AU.AW6, AU.AW11, AU.G3
    Q, part, G = [], {}, {}
    kbs = [ds for ds in KBS if ds in trains or ds in reads]
    if kbs:
        import anchor_kbfit as KF
        KB = KF.KB
        for mod, key in ((KF, "anchor_kbfit"), (KB, "anchor_gen_kb")):
            if AW.sha(Path(mod.__file__)) != AU.KB_PINS[key]:
                raise SystemExit(f"{mod.__file__} is not the pinned file")
    for ds in kbs:
        AW11.STATE.update({"max_len": max_len, "proj": False, "rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0})
        carves = ["x1", "select"] + trains[ds] if ds in trains else [READ[ds]]
        Q0, p0, rec_sha, rel = [], {}, {}, None
        for carve in carves:
            Ql, rs, rl = KF.load_carve(ds, carve)
            if rel is not None and rl != rel:
                raise SystemExit("the carves disagree on the relation bank")
            rel = rl
            p0[carve] = list(range(len(Q0), len(Q0) + len(Ql)))
            Q0.extend(Ql)
            rec_sha.update(rs)
            log(f"{ds}/{carve}: {len(Ql)} rows, {time.time() - t0:.0f}s")
        load_state = dict(AW11.STATE)
        count_rows = [i for lk in trains[ds] for i in p0[lk]] if ds in trains else p0[READ[ds]]
        rank_of, vocab, phi, counts = KB.kb_relations(ds, Q0, count_rows, int(rel["n_relations"]))
        Qv, st = KB.kb_view(Q0, rank_of)
        del Q0
        off = len(Q)
        Q.extend(Qv)
        for lk, rows in p0.items():
            part[(ds, lk)] = [off + i for i in rows]
        G[ds] = {"kind": "kb", "role": "train" if ds in trains else "read", "phi": phi, "phi_sha": KB.RELS[ds][1],
                 "n_rel": len(vocab), "vocab": vocab,
                 "info": {"rel_vocab": KB.RELS[ds][0], "rel_embeddings": KB.RELS[ds][1], "look_score_kb": KB.LOOK_KB_SHA,
                          "look_score_kb_records": rec_sha, "relations_ranked_over": "training rows" if ds in trains else
                          f"the read carve {READ[ds]} (label-free)",
                          "relations": {**rel, "ranked": counts[:64], "n_ranked": len(counts)}, "kb_view": st,
                          "pruned_loader": load_state}}
    for ds in [d for d in AU.PASSAGE if d in trains or d in reads]:
        AW6.rebind(ds)
        carves = ["x1", "select"] + trains[ds] if ds in trains else [READ[ds]]
        Qd, pd, rec_sha, load_state = AG.load_looks(carves, max_len, t0)
        checks, voc = AW.anchor_tables(Qd)
        log(f"{ds}: anchors attached to {len(Qd)} rows: {checks}, {time.time() - t0:.0f}s")
        phi, phi_sha = G3.phrase_table(ds)
        off = len(Q)
        Q.extend(Qd)
        for lk, rows in pd.items():
            part[(ds, lk)] = [off + i for i in rows]
        G[ds] = {"kind": "passage", "role": "train" if ds in trains else "read", "phi": phi, "phi_sha": phi_sha, "n_rel": None,
                 "vocab": list(voc.get("w2", [])),
                 "info": {"phrase_table": phi_sha, "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA, "look_score_records": rec_sha,
                          "look_score_sha": AG.AW3.LOOK_SCORE_SHA, "flag_checks": checks, "pruned_loader": load_state}}
        del Qd
    return Q, part, G


def with_abs(rec, base):
    rec["abs (R@5, FC@5, hit@1)"] = [round(f / g, 4) if g > 1e-9 else None for f, g in zip(rec["fit"], base["gnn0"])]
    return rec


def parse_arms(spec, names, rule, AU, AG, G3):
    """qd_gnn.run's arm grammar, unchanged (a walk variant under rule both is parsed as np: its loss is the loop's)."""
    wr = "np" if rule == "both" else rule
    arms = {}
    for nm in spec.split(","):
        if nm.startswith("QD-"):
            sp = QG.parse_qd(nm)
            if sp["kind"] == "ID" and len(names) > 1:
                raise SystemExit(f"{nm}: a rank's meaning is one graph's; QD-ID fits one graph only")
            arms[nm] = sp
        elif nm.startswith("W:"):
            v = nm[2:]
            if ":" in v:
                raise SystemExit(f"{nm}: a variant names no train looks")
            vv = f"{AU.TAG}:{v}" + ("/lin" if v.count("/") == 0 else "") + ("" if v.count("/") >= 2 else f"/{wr}")
            sp = AG.parse(vv, (AU.TAG,)) if v.startswith("T0") else G3.parse(vv, (AU.TAG,))
            if sp["rule"] != wr:
                raise SystemExit(f"{nm}: rule {sp['rule']} under --rule {rule}")
            if sp.get("pca") or (sp["family"] == "free" and sp["tok"] != "T0"):
                raise SystemExit(f"{nm}: refused as anchor_univ refuses it")
            arms[nm] = {**sp, "family_arm": "walk"}
        else:
            raise SystemExit(f"{nm}: QD-... or W:<variant>")
    return arms


def step_loss2(s, gold, z, rule):
    """qd_gnn.step_loss for np and p; both is their sum (each over its own rows with a gold)."""
    if rule != "both":
        return QG.step_loss(s, gold, z, rule)
    a, b = QG.step_loss(s, gold, z, "np"), QG.step_loss(s, gold, z, "p")
    if a is None and b is None:
        return None
    return (a if a is not None else 0.0) + (b if b is not None else 0.0)


def fit_shared2(arm, names, tr, sel, rule, epochs, lr, wd, seed, balance, AU):
    """qd_gnn.fit_shared, line for line, with step_loss2 and, under rule both, np's select read."""
    torch.manual_seed(seed)
    model = arm.make()
    mats = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    rng = np.random.default_rng(seed)
    streams = {g: {"perm": None, "pos": 0} for g in names}
    best, best_state, best_ep, curve, by_graph = -1.0, None, -1, [], {g: [] for g in names}
    sel_rule = "np" if rule == "both" else rule
    cols = slice(0, 2) if sel_rule == "p" else slice(0, 3)
    for ep in range(epochs):
        t_ep = time.time()
        batches = AU.epoch_batches(rng, names, tr, balance, streams)
        model.train()
        for g, rows in batches:
            arm.prepare(model, g)
            s, gold, z = arm.forward(model, g, rows)
            loss = step_loss2(s, gold, z, rule)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
        sc = {}
        for g in names:
            arm.prepare(model, g)
            sc[g] = float(arm.read(model, g, list(sel[g]), sel_rule)[:, cols].mean())
            by_graph[g].append(round(sc[g], 4))
        score = sc[names[0]] if len(names) == 1 else float(np.mean([sc[g] for g in names]))
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"  ep {ep}: select {round(score, 4)} " + " ".join(f"{g} {sc[g]:.4f}" for g in names) + f" ({time.time() - t_ep:.0f}s)")
    model.load_state_dict(best_state)
    return model, best_ep, curve, by_graph


def score_rows(arm, model, g, rows, TY=None):
    """Each row's (s, z) over its pool, from the arm's forward in eval mode (qd_gnn's read chunks)."""
    model.eval()
    out = []
    qd = isinstance(arm, QG.QDArm)
    step = 128 if qd else 256
    with torch.no_grad():
        for s0 in range(0, len(rows), step):
            rr = rows[s0:s0 + step]
            s, _gold, z = arm.forward(model, g, rr) if qd else arm.forward(model, g, rr, TY)
            for bi, i in enumerate(rr):
                n = arm.Q[i]["n"]
                out.append((s[bi, :n].numpy().copy(), z[bi, :n].numpy().copy()))
    model.train()
    return out


def metrics_at(SZ, rows, Q, margin, metrics_of):
    """qd_gnn.metrics_rows from stored scores: margin None reads unprotected (np), 0.0 places the twin's rank-1 first
    on every row (p), m on the rows where its z margin over rank 2 is at least m."""
    out = []
    for (s, z), i in zip(SZ, rows):
        if margin is not None:
            f = int(np.argmax(z))
            if z.size > 1:
                zz = z.copy()
                zz[f] = -np.inf
                gap = float(z[f] - zz.max())
            else:
                gap = math.inf
            if gap >= margin:
                s = s.copy()
                s[f] = np.inf
        out.append(metrics_of(s, Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def choose_margin(arm, model, names, sel, Q, metrics_of):
    by = {str(m): [] for m in MARGINS}
    for g in names:
        arm.prepare(model, g)
        SZ = score_rows(arm, model, g, list(sel[g]))
        for m in MARGINS:
            by[str(m)].append(float(metrics_at(SZ, list(sel[g]), Q, m, metrics_of).mean()))
    mean = {k: round(float(np.mean(v)), 4) for k, v in by.items()}
    best, best_v = None, -1.0
    for m in MARGINS:
        if mean[str(m)] > best_v + 1e-12:
            best, best_v = m, mean[str(m)]
    return best, mean


def run(a):
    out_path = Path(a.out)
    if a.rule not in RULES:
        raise SystemExit(f"--rule {'|'.join(RULES)}")
    if a.looks_2w == "six":
        Q6.bind()
    elif a.looks_2w != "pilot":
        raise SystemExit("--looks-2w six|pilot")
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    if sha(AU.__file__) != QG.PINS["anchor_univ"]:
        raise SystemExit("anchor_univ.py is not the pinned file")
    AU.check_pins()
    AU.AC.bind()
    AG, G3, A16, AW11 = AU.AG, AU.G3, AU.A16, AU.AW11
    for mod, key in ((AW11, "anchor_walk11"), (A16, "l16_look_analyze")):
        if sha(mod.__file__) != QG.PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    if tuple(AU.AW6.MARGINS) != MARGINS:
        raise SystemExit("MARGINS differ from anchor_walk6.MARGINS")
    QG.bind_frontier(AW11)
    trains = AU.parse_train(a.train)
    names = list(trains)
    reads = a.read.split(",") if a.read else []
    for r in reads:
        if r not in READ:
            raise SystemExit(f"--read {r}: one of {list(READ)}")
        if r in trains:
            raise SystemExit(f"--read {r}: a training graph is read on its own x1 half B already")
    if len(set(reads)) != len(reads):
        raise SystemExit("--read: each graph once")
    allg = names + reads
    arms = parse_arms(a.arms, names, a.rule, AU, AG, G3)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    max_len = max(sp["max_len"] for sp in arms.values())
    Q, part, G = load_all(trains, reads, max_len, t0, AU)
    log(f"loaded {len(Q)} rows at max_len {max_len}: " + ", ".join(f"{g} {G[g]['info']['pruned_loader']}" for g in allg))
    for i in range(0, len(Q), max(1, len(Q) // 200)):
        if not np.array_equal(QG.famdir(Q[i]), A16.famdir(Q[i])):
            raise SystemExit("famdir differs from l16_look_analyze.famdir")
    TOK = [QG.edge_labels(q) for q in Q]
    tr = {g: [i for lk in trains[g] for i in part[(g, lk)]] for g in names}
    sel = {g: part[(g, "select")] for g in names}
    B = {g: part[(g, "x1")][1::2] for g in names}
    for g in reads:
        B[g] = list(part[(g, READ[g])]) if g in WHOLE else part[(g, READ[g])][1::2]
    RB = {g: AG.Reader(Q, B[g], QG.READ_SEED) for g in allg}
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "qd_gnn2", "train": trains, "read": reads, "rule": a.rule, "looks_2w": a.looks_2w, "arms": list(arms),
           "script_sha256": sha(__file__), "pins": {**QG.PINS, "qd_gnn": Q6.QD_SHA, "qd_six": sha(Q6.__file__), "look_x_six": Q6.LX_SHA},
           "epochs": a.epochs, "lr": a.lr, "wd": a.wd, "balance": a.balance, "threads": a.threads, "max_len": max_len,
           "margins": [str(m) for m in MARGINS],
           "graphs": {g: {"role": G[g]["role"], "kind": G[g]["kind"], "train_rows": len(tr.get(g, [])), "select_rows": len(sel.get(g, [])),
                          "B_rows": len(B[g]), "B_carve": "x1 half B" if g not in WHOLE else f"{READ[g]} whole",
                          "info": G[g]["info"], "B": RB[g].base()} for g in allg},
           "results": {}}
    per_row = {}
    model_dir = out_path.parent / (out_path.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, **{f"B_rows_{g}": np.asarray(B[g]) for g in allg})

    for ai, (name, sp) in enumerate(arms.items()):
        t1 = time.time()
        qd = sp["family"] == "qd"
        if qd:
            arm = QG.QDArm(sp, Q, TOK, G, z_of, A16, sp["K"] if sp["kind"] == "ID" else 0)
            alt = {}
        else:
            K = sp["K"]
            TY, nt = [None] * len(Q), None
            for g in allg:
                rows_g = sorted({i for (gg, _lk), rr in part.items() if gg == g for i in rr})
                nt, ty = AG.types_for(Q, sp, None if sp["tok"] == "T0" else AG.identity_map(K), rows_g)
                for i in rows_g:
                    TY[i] = ty[i]
                del ty
            arm = QG.WalkArm(sp, Q, TY, nt, G, z_of, AU, AG, G3, A16)
            arm.make_fn = G3.make_for(sp, nt, G[names[0]]["phi"], None)
            alt = {g: ({"NR": AG.types_for(Q, sp, np.full(QG.R_TOP + 1, K, dtype=np.int64), B[g])[1]} if sp["tok"] == "A" else {})
                   for g in allg}
        label_arm = (qd and sp["kind"] != "T0") or (not qd and sp["tok"] == "A")
        text_arm = (qd and sp["kind"] == "TXT") or (not qd and sp["tok"] == "A" and sp["family"] != "free")
        seeds_out = {}
        for sd in [int(x) for x in a.seeds.split(",")]:
            t2 = time.time()
            model, best_ep, curve, by_graph = fit_shared2(arm, names, tr, sel, a.rule, a.epochs, a.lr, a.wd, sd, a.balance, AU)
            margin, by_margin = choose_margin(arm, model, names, sel, Q, A16.metrics_of)
            rd = {}
            for g in allg:
                R, base = RB[g], RB[g].base()
                id_ok = not (G[g]["role"] == "read" and qd and sp["kind"] == "ID")
                arm.prepare(model, g)
                e, mID = {}, {}
                if id_ok:
                    SZ = score_rows(arm, model, g, B[g])
                    for rl, mg in (("np", None), ("p", 0.0), ("mp", margin)):
                        mID[rl] = metrics_at(SZ, B[g], Q, mg, A16.metrics_of)
                        per_row[f"{ai}_{sd}_{g}_ID_{rl}"] = mID[rl]
                        e[f"ID/{rl}"] = with_abs(R.record(mID[rl], by_type=(rl != "p")), base)
                    del SZ
                if label_arm:
                    if qd:
                        arm.nr = True
                        SZ = score_rows(arm, model, g, B[g])
                        arm.nr = False
                    else:
                        SZ = score_rows(arm, model, g, B[g], alt[g]["NR"])
                    for rl, mg in (("np", None), ("p", 0.0), ("mp", margin)):
                        m_alt = metrics_at(SZ, B[g], Q, mg, A16.metrics_of)
                        per_row[f"{ai}_{sd}_{g}_NR_{rl}"] = m_alt
                        e[f"NR/{rl}"] = with_abs(R.record(m_alt, by_type=False), base)
                        if rl in mID:
                            e[f"NR - ID/{rl}"] = AU.AW3.boot_pair(m_alt - mID[rl], R.W)
                    del SZ
                if G[g]["kind"] == "kb" and text_arm and id_ok:
                    phi = G[g]["phi"]
                    phi_s = phi.copy()
                    k_ = min(G[g]["n_rel"], QG.R_TOP)
                    phi_s[:k_] = phi[(np.arange(k_) + 1) % k_]
                    arm.prepare(model, g, phi_s)
                    m_alt = metrics_at(score_rows(arm, model, g, B[g]), B[g], Q, None, A16.metrics_of)
                    arm.prepare(model, g)
                    per_row[f"{ai}_{sd}_{g}_SHUF_np"] = m_alt
                    e["SHUF/np"] = with_abs(R.record(m_alt, by_type=False), base)
                    e["SHUF - ID/np"] = AU.AW3.boot_pair(m_alt - mID["np"], R.W)
                rd[g] = e
            pt = model_dir / f"a{ai}_s{sd}.pt"
            torch.save({"arm": name, "spec": sp, "state_dict": model.state_dict(), "train": trains, "rule": a.rule, "margin": margin,
                        "phi_sha": {g: G[g]["phi_sha"] for g in names}}, pt)
            seeds_out[str(sd)] = {"best_epoch": best_ep, "curve": curve, "curve_by_graph": by_graph, "margin": str(margin),
                                  "select_by_margin": by_margin, "reads": rd,
                                  "model_file": str(pt.relative_to(QG.ROOT)) if pt.is_relative_to(QG.ROOT) else str(pt),
                                  "seconds": round(time.time() - t2, 1)}
            log(f"{name}#{sd}: ep {best_ep}, margin {margin}; " + "; ".join(
                f"{g}{'*' if G[g]['role'] == 'read' else ''} " + ", ".join(
                    f"{k} rho {v['rho (R@5, FC@5, hit@1)']} abs {v['abs (R@5, FC@5, hit@1)']}"
                    for k, v in rd[g].items() if k in ("ID/np", "ID/mp", "NR/np"))
                for g in allg))
            res["results"][name] = {"spec": {k: v for k, v in sp.items() if k != "seeds"}, "params": int(sum(p.numel() for p in model.parameters())),
                                    "seeds_read": seeds_out, "seconds": round(time.time() - t1, 1)}
            save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--read", default="")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", default="np")
    ap.add_argument("--looks-2w", default="six")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--balance", default="graph")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=2)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
