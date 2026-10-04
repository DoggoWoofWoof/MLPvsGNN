"""Design look (untracked; not a result and not filed): relation models fitted on a knowledge graph's own carves, read
on relations the fit never saw, with no relations, with the relations' meanings shuffled and with their order swapped.

Every relation model so far was fitted on a passage graph (2wiki, hotpotqa, musique: anchor phrases as edge labels) and
read on metaqa zero-shot, and each one harms there at every kappa (ksweep_kb.py). On metaqa a typed walk's relation ORDER
is worth +0.269 rho in level 8 (configs/mp_approx_l8.yaml: rotary TP against the order-blind TP-bag), while on 2wiki and
hotpotqa no operator reads a SWAP effect beyond +-0.24 points (anchor_order.py). This look fits the anchor_gen /
anchor_gen3 / anchor_cos model families on the KB itself: the GNN's own fit carve (metaqa: 5,883 rows, the six pair's
labels) trains, its select carve (1,482) chooses the epoch, carve x1's half B (2,980 rows) is read, as every KB read.
Relations are the KB's names (metaqa: 9), ranked by their count over the fit rows' structural edges (label-free), one
walk edge per stored relation (anchor_gen_kb.kb_view). text, gen and cos models read a relation through its gte-Qwen2
vector (the pinned outputs/m3b/relations/<ds>_rel_embeddings.npy); a free model has a vector per rank; T0 is the
family x direction model.
Reads on B, each against the look's twin and GNN and paired against the main read:
    ID      the fit's own view (with --hold_rel: DROP, the held relations' walk edges deleted, as in training)
    NR      every relation 'other': the untyped KB (with --mres, the model's own untyped component)
    SHUF    every relation shown with the next relation's text (free: the next one's vector): what the identity adds
    SWAP    (two-edge models) every two-edge walk code (b, t1, t2) read as (b, t2, t1) (anchor_order.swap_types): what
            ORDER adds; exactly 0 for the commutative operators (gtrans, grope)
  with --hold_rel 'R1;R2': the held relations' walk edges are deleted in training and selection, so the model meets no
  trace of them, typed or untyped (an unseen relation), and at the read they are
    REV     shown with their own relation (text, gen, cos: by their text; free: their rank's untrained vector)
    MASK    shown as 'other' (meaningful with --mres, where 'other' is the untyped component)
    FULL    (T0) shown, untyped
  on every B row and on the held rows: the B rows where a walk through a held relation's edge reaches a gold (one-edge
  walks, and walks of up to the loaded length).
Options: --mres (anchor_mres.double: every walk edge gets an untyped copy), --rdrop_* (anchor_gen2's relation dropout),
--kd (anchor_walk8's teacher). The fit loop, loss, rules, selection and reads are anchor_walk8.fit_read_kd and
anchor_walk6.read_rule, imported unchanged. Models are saved as anchor_gen3 saves them (dataset the KB, vocab_w2 its
relation names in rank order, phi_sha its relation table's sha, 'hold' None), so mode xread reads them zero-shot on a
passage graph through anchor_gen3's cross read (anchor_gen's for T0), with anchor_cos bound and, for --mres models,
anchor_mres installed. Nothing reads a neighbour's score or state.

    python outputs/mp_approx_kb_anchor/host/anchor_kbfit.py fit --variants fit:A256-1/txt0,fit:A256/gropet --out .../anchor_kbfit_x.json
    python outputs/mp_approx_kb_anchor/host/anchor_kbfit.py fit --variants fit:A256-1/cos --hold_rel "written by" --out ...
    python outputs/mp_approx_kb_anchor/host/anchor_kbfit.py xread --target 2wiki --models a.pt,b.pt --out .../anchor_kbfit_xread_2w.json
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
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_2wiki_anchor" / "host"))
sys.path.insert(0, str(HERE))
import anchor_gen_kb as KB  # noqa: E402
import anchor_cos as AC  # noqa: E402
import anchor_order as AO  # noqa: E402
import anchor_mres as MR  # noqa: E402

G3 = KB.G3
AG, G2N = G3.AG, G3.G2N
AW11, AW8, AW6, A16, AW, AW3 = AG.AW11, AG.AW8, AG.AW6, AG.A16, AG.AW, AG.AW3
R_TOP = AG.R_TOP
PINS = {"anchor_gen_kb": "4e09d59de0a60767eb591863352389ff99e164f0e2cd841b858debd309ee7952",
        "anchor_cos": "1039cf2f043bc6e3c5d342d0ea8f8f89ee3a1873c6e78a94f4078e99b7f667bf",
        "anchor_order": "467d215b4d6c25f28ff912b2f08b97c67e28ccb5277d066a33e189ed243baea2",
        "anchor_mres": "e9dcc06e043bd590c14b5586adcdefe94eb8745a1444f1f3d52d122fbc8b93fd",
        "anchor_gen3": "ee5f76dc7d391649e04ab2b3575d4fdd30dd62cd2a34b1b7dbd114d56f7b14a7",
        "anchor_gen": "bdc938f546cff1192dda82701e92cf7003fce8aded07831c2c7930119ce4f108",
        "anchor_gen2": "cbb976408648e2da0d5398df7310b2695a471e246f7def05413f74cec84711c6"}
TRAIN, SELECT, READ = "fit", "select", "x1"
log = AG.log


def check_pins():
    for mod, key in ((KB, "anchor_gen_kb"), (AC, "anchor_cos"), (AO, "anchor_order"), (MR, "anchor_mres"), (G3, "anchor_gen3"),
                     (AG, "anchor_gen"), (G2N, "anchor_gen2")):
        if AW.sha(Path(mod.__file__)) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    AG.check_pins()


def load_carve(ds, carve):
    """One carve of the KB's look (anchor_gen_kb.load_look for any carve): every record written by the pinned
    look_score_kb.py without --limit, every shard present, one relation bank."""
    look = KB.LOOK_ROOT / ds / carve
    recs = sorted(look.glob("record*.json"))
    if not recs:
        raise SystemExit(f"{look} has no record")
    rec_sha, rel, shards = {}, None, set()
    for r in recs:
        rj = json.loads(r.read_text(encoding="utf-8"))
        if rj.get("script_sha256") != KB.LOOK_KB_SHA:
            raise SystemExit(f"{r} was not written by the pinned look_score_kb.py")
        if rj.get("dataset") != ds or rj.get("carve") != carve:
            raise SystemExit(f"{r} is not {ds} carve {carve}")
        if rj.get("limit") is not None:
            raise SystemExit(f"{r} is a --limit look")
        if rel is not None and rj["relations"] != rel:
            raise SystemExit("the shards disagree on the relation bank")
        rel = rj["relations"]
        rec_sha[f"{carve}/{r.name}"] = rj["script_sha256"]
        if rj.get("shard"):
            shards.add(tuple(rj["shard"]))
    if shards and {s[0] for s in shards} != set(range(next(iter(shards))[1])):
        raise SystemExit(f"{carve}: shards missing: {sorted(shards)}")
    _ids, Q = KB.load_pruned_kb(look)
    return Q, rec_sha, rel


def walk_rank(q):
    """Each structural walk edge's relation rank (kb_view stores it in every stored direction, -1 elsewhere)."""
    return np.maximum(q["w2_f"], q["w2_b"]).astype(np.int64)


def drop_held(q, held_ranks):
    """(q without the walk edges of the held relations, how many went). A KB view row has no other edge carrying them."""
    hit = np.isin(walk_rank(q), held_ranks)
    if not hit.any():
        return q, 0
    keep = np.ones(q["u"].size, dtype=bool)
    keep[np.flatnonzero(q["fam"] == 0)[hit]] = False
    nq = dict(q)
    for k in ("u", "v", "fam", "fwd", "bwd", "w"):
        nq[k] = q[k][keep]
    for k in ("w2_f", "w2_b"):
        nq[k] = q[k][~hit]
    return nq, int(hit.sum())


def held_rows(Q, TY, nt, K, rows, held_ranks, one_edge=False):
    """The rows with a walk type through a held relation's edge whose reach set holds a gold. A code is
    (b tb + t1 + 1) tb + (t2 + 1, 0 for one edge), tb = nt + 1; a structural token is d (K + 1) + rank."""
    tb = nt + 1
    hs = {int(h) for h in held_ranks}
    out = []
    for i in rows:
        g = Q[i]["gold"]
        for c, R in TY[i].items():
            t2p, t1p = c % tb, (c // tb) % tb
            if one_edge and t2p:
                continue
            toks = [t - 1 for t in (t1p, t2p) if t > 0]
            if any(t < 3 * (K + 1) and t % (K + 1) in hs for t in toks) and g[np.asarray(R, dtype=np.int64)].any():
                out.append(i)
                break
    return out


def parse(name):
    base = name.partition(":")[2].split("/")[0]
    sp = AG.parse(name, (TRAIN,)) if base.startswith("T0") else G3.parse(name, (TRAIN,))
    if sp.get("pca"):
        raise SystemExit(f"{name}: a PCA code is fitted on the vocabulary's own spread, which a KB's few relations do not give")
    return sp


def run_fit(a):
    out_path = Path(a.out)
    if a.mres and "mres" not in out_path.name:
        raise SystemExit("--mres: --out must carry 'mres' in its file name (anchor_mres reads these models)")
    torch.set_num_threads(2)
    t0 = time.time()
    AC.bind()
    specs = {name: parse(name) for name in a.variants.split(",")}
    hold_names = [s for s in (a.hold_rel or "").split(";") if s]
    max_len = max(sp["max_len"] for sp in specs.values())
    AW11.STATE.update({"max_len": max_len, "proj": False, "rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0})
    Q0, part, rec_sha, rel = [], {}, {}, None
    for carve in (READ, SELECT, TRAIN):
        Ql, rs, rl = load_carve(a.dataset, carve)
        if rel is not None and rl != rel:
            raise SystemExit("the carves disagree on the relation bank")
        rel = rl
        part[carve] = list(range(len(Q0), len(Q0) + len(Ql)))
        Q0.extend(Ql)
        rec_sha.update(rs)
        log(f"look {a.dataset}/{carve}: {len(Ql)} rows, {time.time() - t0:.0f}s")
    load_state = dict(AW11.STATE)
    rank_of, vocab_w2, phi, counts = KB.kb_relations(a.dataset, Q0, part[TRAIN], int(rel["n_relations"]))
    n_rel = len(vocab_w2)
    phi_sha = KB.RELS[a.dataset][1]
    for nm in hold_names:
        if nm not in vocab_w2:
            raise SystemExit(f"--hold_rel {nm!r} is not one of {vocab_w2}")
    held_ranks = np.asarray(sorted(vocab_w2.index(nm) for nm in hold_names), dtype=np.int64)
    Qv, kb_st = KB.kb_view(Q0, rank_of)
    del Q0
    x1 = part[READ]
    B_rows = x1[1::2]
    Qf, drop_st = None, None
    if held_ranks.size:
        Qf = [None] * len(Qv)
        for i in x1:
            Qf[i] = Qv[i]
        nd = 0
        for i in range(len(Qv)):
            Qv[i], k = drop_held(Qv[i], held_ranks)
            nd += k
        drop_st = {"walk_edges_deleted": nd}
        left = sum(int(np.isin(walk_rank(Qv[i]), held_ranks).sum()) for i in range(len(Qv)))
        if left:
            raise SystemExit("a held relation's walk edge survived the deletion")
    if a.mres:
        for q in Qv:
            MR.double(q)
        if Qf is not None:
            for i in x1:
                MR.double(Qf[i])
    RB, RX = AG.Reader(Qv, B_rows, 20261002), AG.Reader(Qv, x1, 20261003)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Qv]
    zT_of = [None] * len(Qv)
    if a.kd > 0:
        for i in part[TRAIN]:
            zT_of[i] = AW8.teacher_of(Qv[i], a.teacher)
    phi_s = phi.copy()
    k_ = min(n_rel, R_TOP)
    phi_s[:k_] = phi[(np.arange(k_) + 1) % k_]
    hrows = {}
    if held_ranks.size:
        ref = {"tok": "A", "K": 256, "max_len": max_len}
        nt_r, TYr = AG.types_for(Qf, ref, AG.identity_map(256), B_rows)
        hrows["held_rows_1edge"] = held_rows(Qf, TYr, nt_r, 256, B_rows, held_ranks, one_edge=True)
        if max_len == 2:
            hrows["held_rows"] = held_rows(Qf, TYr, nt_r, 256, B_rows, held_ranks)
        del TYr
    main_key = "DROP" if held_ranks.size else "ID"
    rdrop = {"row": a.rdrop_row, "edge": a.rdrop_edge, "unit": a.rdrop_unit, "seed": G2N.DROP_SEED}
    res = {"look": "anchor_kbfit", "mode": "fit", "dataset": a.dataset, "script_sha256": AW.sha(Path(__file__)),
           "pins": {**PINS, "look_score_kb": KB.LOOK_KB_SHA, "look_score_kb_records": rec_sha, "rel_vocab": KB.RELS[a.dataset][0],
                    "rel_embeddings": KB.RELS[a.dataset][1]},
           "relations": {**rel, "ranked_by_fit_rows": counts}, "kb_view": kb_st, "hold_rel": hold_names, "held_ranks": held_ranks.tolist(),
           "drop": drop_st, "mres": bool(a.mres), "mres_doubled": dict(MR.STATS) if a.mres else None, "pruned_loader": load_state,
           "looks": {lk: len(r) for lk, r in part.items()}, "epochs": a.epochs, "lr": a.lr, "wd": a.wd,
           "kd": {"ce": a.ce, "kd": a.kd, "T": a.T, "teacher": a.teacher} if a.kd > 0 else None, "rdrop": rdrop,
           "B": RB.base(), "x1": RX.base(), "held_rows": {k: len(v) for k, v in hrows.items()}, "variants": {}}
    if held_ranks.size:
        res["held_rows_base"] = {k: RB.subset(RB.t, v) for k, v in hrows.items()}
    per_row = {}
    model_dir = out_path.parent / (out_path.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    pos = {i: j for j, i in enumerate(B_rows)}

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))

    cache = {}
    for name, sp in specs.items():
        t1 = time.time()
        K = sp["K"]
        tr = part[TRAIN]
        key = (sp["tok"], K, sp["max_len"])
        if key not in cache:
            cache.clear()
            if sp["tok"] == "T0":
                nt, TY = AG.types_for(Qv, sp, None)
                stats = None
            else:
                drops, whole = G2N.drop_masks(Qv, tr, K, a.rdrop_row, a.rdrop_edge, a.rdrop_unit, G2N.DROP_SEED)
                nt, TY = G2N.types_dropped(Qv, sp, AG.identity_map(K), drops)
                stats = {"rows_whole": whole, "rows_partial": len(drops) - whole, "edges_dropped": int(sum(int(dm.sum()) for dm in drops.values()))}
            cache[key] = (nt, TY, stats)
        nt, TY, stats = cache[key]
        alt = {}
        if sp["tok"] == "A":
            Qa = Qf if Qf is not None else Qv
            alt["NR"] = (Qa, AG.types_for(Qa, sp, np.full(R_TOP + 1, K, dtype=np.int64), B_rows)[1])
            if sp["family"] == "free":
                alt["SHUF"] = (Qv, AG.types_for(Qv, sp, KB.shuffled(AG.identity_map(K), n_rel, K), B_rows)[1])
            if held_ranks.size:
                alt["REV"] = (Qf, AG.types_for(Qf, sp, AG.identity_map(K), B_rows)[1])
                rm_mask = AG.identity_map(K).copy()
                rm_mask[held_ranks] = K
                alt["MASK"] = (Qf, AG.types_for(Qf, sp, rm_mask, B_rows)[1])
        elif held_ranks.size:
            alt["FULL"] = (Qf, AG.types_for(Qf, sp, None, B_rows)[1])
        if sp["max_len"] == 2:
            alt["SWAP"] = (Qv, AO.swap_types(TY, nt, B_rows))
        make = G3.make_for(sp, nt, phi, None)
        models, seeds_out = [], {}
        for sd in sp["seeds"]:
            t2 = time.time()
            model, reads, best_ep, curve, margin, by_margin = AW8.fit_read_kd(Qv, TY, nt, tr, part[SELECT], {"B": B_rows, "x1": x1}, z_of, zT_of,
                                                                              make, a.epochs, sd, sp["rule"], a.lr, a.wd, a.ce, a.kd, a.T)
            models.append(model)
            vi = len(res["variants"])
            mk = f"{vi}_{sd}_{main_key}"
            rd = {main_key: RB.record(reads["B"]), main_key + "_x1": RX.record(reads["x1"], by_type=False)}
            per_row[mk] = reads["B"]
            got = {}
            for nm, (Qa_, TYa) in alt.items():
                got[nm] = AG.read(model, Qa_, TYa, nt, B_rows, z_of, margin)
            if sp["tok"] == "A" and sp["family"] != "free":
                G3.set_phi(model, phi_s, K)
                got["SHUF"] = AG.read(model, Qv, TY, nt, B_rows, z_of, margin)
                G3.set_phi(model, phi, K)
            for nm, m_alt in got.items():
                per_row[f"{vi}_{sd}_{nm}"] = m_alt
                rd[nm] = RB.record(m_alt)
                rd[f"{nm} - {main_key}"] = AW3.boot_pair(m_alt - reads["B"], RB.W)
            if hrows:
                hr = {}
                for tag, rows_ in hrows.items():
                    hx = np.asarray([pos[i] for i in rows_], dtype=np.int64)
                    ent = {main_key: RB.subset(reads["B"], rows_)}
                    for nm in ("REV", "MASK", "FULL", "NR"):
                        if nm in got and hx.size:
                            ent[nm] = RB.subset(got[nm], rows_)
                            ent[f"{nm} - {main_key}"] = AW3.boot_pair((got[nm] - reads["B"])[hx], RB.W[:, hx])
                    hr[tag] = ent
                rd["held"] = hr
            pt = model_dir / f"v{vi}_s{sd}.pt"
            G3.save_model(pt, model, sp, nt, margin, a.dataset, vocab_w2, phi_sha,
                          {"hold": None, "hold_rel": hold_names, "name": name, "seed": sd, "rdrop": rdrop, "mres": bool(a.mres), "kd": res["kd"]})
            seeds_out[str(sd)] = {"best_epoch": best_ep, "curve": curve, "margin": None if margin is None else str(margin), "by_margin_select": by_margin,
                                  "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
                                  "gam": model.gam.detach().round(decimals=4).tolist() if hasattr(model, "gam") else None,
                                  "model_file": str(pt.relative_to(ROOT)) if pt.is_relative_to(ROOT) else str(pt), "reads": rd,
                                  "seconds": round(time.time() - t2, 1)}
            log(f"{name}#{sd} ({sp['family']}): ep {best_ep}, " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in rd.items()
                                                                       if isinstance(v, dict) and "rho (R@5, FC@5, hit@1)" in v))
        m0 = models[0]
        v = {**sp, "train_rows": len(tr), "tokens": nt, "types_per_row": round(float(np.mean([len(TY[i]) for i in tr])), 1),
             "params": int(sum(p.numel() for p in m0.parameters())), "params_used": AG.params_used(m0, sp), "rdrop_stats": stats,
             "order_rows_B": len(AO.order_rows(TY, nt, B_rows)) if sp["max_len"] == 2 else None, "seeds_read": seeds_out}
        v["seconds"] = round(time.time() - t1, 1)
        res["variants"][name] = v
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


def run_xread(argv):
    """KB-fitted models read zero-shot on a passage graph's carve x1: anchor_gen3's cross read (anchor_gen's for T0),
    with anchor_cos bound and, for --mres models, anchor_mres installed."""
    out = argv[argv.index("--out") + 1] if "--out" in argv else ""
    mp = argv[argv.index("--models") + 1] if "--models" in argv else ""
    if "kbfit" not in Path(out).name:
        raise SystemExit("xread: --out must carry 'kbfit' in its file name")
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in mp.split(",") if p]
    if not cks or any(ck["dataset"] not in KB.RELS for ck in cks):
        raise SystemExit("xread: --models must be KB-fitted models")
    mres = {bool(ck.get("mres")) for ck in cks}
    t0m = {ck["spec"]["tok"] == "T0" for ck in cks}
    if len(mres) != 1 or len(t0m) != 1:
        raise SystemExit("xread: one call reads models of one view (all --mres or none) and one kind (all T0 or none)")
    if mres == {True}:
        if "mres" not in Path(out).name:
            raise SystemExit("xread: --mres models need 'mres' in --out")
        MR.install()
    AC.bind()
    if t0m == {True}:
        AG.main(["--xread"] + argv)
    else:
        G3.main(["--xread"] + argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("fit", "xread"):
        raise SystemExit("anchor_kbfit.py fit|xread ...")
    mode, rest = argv[0], argv[1:]
    check_pins()
    if mode == "xread":
        run_xread(rest)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metaqa", choices=tuple(KB.RELS))
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ce", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--teacher", default="gm", choices=AW8.TEACHERS)
    ap.add_argument("--hold_rel", default=None)
    ap.add_argument("--mres", action="store_true")
    ap.add_argument("--rdrop_row", type=float, default=0.0)
    ap.add_argument("--rdrop_edge", type=float, default=0.0)
    ap.add_argument("--rdrop_unit", default="edge", choices=("edge", "phrase"))
    a = ap.parse_args(rest)
    if a.ce < 0 or a.kd < 0 or a.ce + a.kd <= 0 or a.T <= 0:
        raise SystemExit("ce and kd must be non-negative, not both 0, and T positive")
    if "kbfit" not in Path(a.out).name:
        raise SystemExit("--out must carry 'kbfit' in its file name")
    run_fit(a)


if __name__ == "__main__":
    main()
