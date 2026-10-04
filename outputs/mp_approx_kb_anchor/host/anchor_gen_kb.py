"""Design look (untracked; not a result and not filed): relation models fitted on a passage graph, read zero-shot on a
knowledge graph through the KB's own relation names.

anchor_gen, anchor_gen2 and anchor_gen3 fit a relation model on one passage dataset's anchor phrases (the two words
before a link) and read it there and, zero-shot, on another passage dataset. A KB's structural edges are facts
(s, r, o): metaqa's 9 relations are named 'directed by', 'written by', 'starred actors', 'has genre', 'has tags',
'in language', 'release year', 'has imdb rating' and 'has imdb votes' (outputs/m3b/relations/<ds>_rel_vocab.json, embedded
by the pipeline that embedded the anchor phrases: gte-Qwen2-1.5B, document mode, 64 tokens, bfloat16, unit-L2, float16).
This look reads saved models on a KB's carve x1 (half B, as every round), scored by look_score_kb.py (the six pair's twin
and GNN, the families the passage looks read, and each structural edge's relation ids):
    XD       the KB's relations as edge labels. A structural message edge u -> v whose pair stores relations R is one
             walk edge per r in R (a pair holding several facts is reachable through each), its direction token the
             look's (u -> v stored, v -> u stored, both), so a fact (u, r, v) walked from u is r forward and walked from v
             is r backward, as an anchor on u -> v is. Relations are ranked by their count over x1's structural edges
             (label-free; ties by id). text and gen models read r's text vector; a free model matches r's text to its
             own phrase strings, else 'other'.
    XD_1R    each pair keeps one relation, its most frequent (a passage edge has one anchor): what the copies add.
    XD_SHUF  every relation shown with the next relation's text (free: the next one's string match), the partition of
             the edges kept: what the relation's meaning adds over the partition alone.
    XD_NR    every relation replaced by 'other': the untyped KB.
A pair stored in both directions holds both directions' relations in its slots (look_score_kb keeps no per-slot
direction); those edges read as 'both' and are counted. Nothing reads a neighbour's score or state; the twin and GNN
numbers are the look's.

    python outputs/mp_approx_kb_anchor/host/anchor_gen_kb.py --target metaqa --models a.pt,b.pt --out PATH
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
import anchor_gen3 as G3  # noqa: E402

AG = G3.AG
AW11, A16, AW, AW3 = AG.AW11, AG.A16, AG.AW, AG.AW3
R_TOP = AG.R_TOP
ANCHOR_GEN3_SHA = "ee5f76dc7d391649e04ab2b3575d4fdd30dd62cd2a34b1b7dbd114d56f7b14a7"
LOOK_KB_SHA = "c635de3d24175bac37da9681f91f467a5f4419cd70c601b68014f97fe8d9ab05"   # look_score_kb.py, the scorer of the looks read here
LOOK_ROOT = HERE / "look"
REL_DIR = ROOT / "outputs" / "m3b" / "relations"
RELS = {"metaqa": ("000bddbe799a60cf0c4e806e0819a9822bc520c2a5d55a4c0b6a669e1b779d11",
                   "752964ecfae44c7e4ebf10468d7d0241cca53b0fdb2488af04ec30deed38b4af"),
        "webqsp": ("ca1bb2825939ec0ada7e6a85ebebfb0698b5b92928f65715f4492ba8cf88c8c3",
                   "58df589aa77c9fa670e51af9fd2cac3ac61033667b4e149ef64c255514a14bd9")}
log = AG.log


def load_pruned_kb(look):
    """anchor_walk11.load_pruned (no projections), with each kept edge's relation slots kept by the same frontier mask."""
    look = Path(look)
    ids = json.loads((look / "ids.json").read_text(encoding="utf-8"))
    info = json.loads((look / "info.json").read_text(encoding="utf-8"))
    Q = [None] * len(ids)
    paths = sorted((look / "chunks").glob("c*.npz"))
    all_rows = np.sort(np.concatenate([np.load(p)["chunk_rows"] for p in paths]))
    if all_rows.size != len(ids) or np.unique(all_rows).size != all_rows.size:
        raise SystemExit("the chunks do not hold each scored row once")
    AW11.check_chunk(paths[0], AW11.STATE["max_len"])
    for p in paths:
        with np.load(p) as zf:
            z = {key: zf[key] for key in zf.files}
        if "e_rel" not in z:
            raise SystemExit(f"{p}: no relation slots (not a look_score_kb chunk)")
        npool, ned = z["q_pool_size"], z["q_edges"]
        po, eo = np.r_[0, np.cumsum(npool)], np.r_[0, np.cumsum(ned)]
        for i, row in enumerate(z["chunk_rows"]):
            a, b = po[i], po[i + 1]
            c, d = eo[i], eo[i + 1]
            n = int(npool[i])
            u, v = z["e_u"][c:d].astype(np.int64), z["e_v"][c:d].astype(np.int64)
            seeds = z["q_seed_local"][i].copy()
            m = AW11.keep_mask(u, v, seeds, n, AW11.STATE["max_len"])
            AW11.STATE["rows"] += 1
            AW11.STATE["edges_in"] += int(u.size)
            AW11.STATE["edges_kept"] += int(m.sum())
            Q[int(np.searchsorted(all_rows, row))] = {
                "n": n, "gt": int(z["q_gold_total"][i]), "metrics": z["q_metrics"][i].copy(), "qemb": z["q_emb"][i].copy(),
                "seeds": seeds, "bucket": z["q_seed_bucket"][i].copy(), "pool": z["pool"][a:b].copy(),
                "score": z["score"][a:b].copy(), "gold": z["is_gold"][a:b].copy(), "proj": np.zeros((n, 0), dtype=np.float32),
                "u": u[m], "v": v[m], "fam": z["e_fam"][c:d][m].astype(np.int64),
                "fwd": z["e_fwd"][c:d][m].astype(bool), "bwd": z["e_bwd"][c:d][m].astype(bool), "w": z["e_w"][c:d][m].copy(),
                "rel": z["e_rel"][c:d][m].astype(np.int64)}
        del z
    missing = [i for i, q in enumerate(Q) if q is None]
    if missing:
        raise SystemExit(f"{len(missing)} rows missing from the chunks")
    for q, inf in zip(Q, info):
        q["type"] = inf["type"]
    return ids, Q


def load_look(ds, max_len, t0):
    """Carve x1 of the KB's look, every record written by the pinned look_score_kb.py, every shard present."""
    look = LOOK_ROOT / ds / "x1"
    recs = sorted(look.glob("record*.json"))
    if not recs:
        raise SystemExit(f"{look} has no record")
    rec_sha, rel, shards = {}, None, set()
    for r in recs:
        rj = json.loads(r.read_text(encoding="utf-8"))
        if rj.get("script_sha256") != LOOK_KB_SHA:
            raise SystemExit(f"{r} was not written by the pinned look_score_kb.py")
        if rj.get("dataset") != ds or rj.get("carve") != "x1":
            raise SystemExit(f"{r} is not {ds} carve x1")
        if rel is not None and rj["relations"] != rel:
            raise SystemExit("the shards disagree on the relation bank")
        rel = rj["relations"]
        rec_sha[r.name] = rj["script_sha256"]
        if rj.get("shard"):
            shards.add(tuple(rj["shard"]))
    if shards and {s[0] for s in shards} != set(range(next(iter(shards))[1])):
        raise SystemExit(f"shards missing: {sorted(shards)}")
    AW11.STATE.update({"max_len": max_len, "proj": False, "rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0})
    _ids, Q = load_pruned_kb(look)
    log(f"look {ds}/x1: {len(Q)} rows, {time.time() - t0:.0f}s")
    return Q, rec_sha, rel, dict(AW11.STATE)


def kb_relations(ds, Q, rows, n_rel):
    """The KB's relation names and text vectors in rank order (their count over the rows' structural edges, ties by
    id); rank_of[relation id] is its rank; phi is R_TOP rows (zero past the vocabulary, never read)."""
    vp, ep = REL_DIR / f"{ds}_rel_vocab.json", REL_DIR / f"{ds}_rel_embeddings.npy"
    if AW.sha(vp) != RELS[ds][0] or AW.sha(ep) != RELS[ds][1]:
        raise SystemExit(f"{ds}'s relation vocab or embeddings are not the pinned files")
    texts = [str(t) for t in json.loads(vp.read_text(encoding="utf-8"))["texts"]]
    emb = np.load(ep).astype(np.float32)
    if emb.shape != (len(texts), 1536) or len(texts) != n_rel:
        raise SystemExit("the relation table does not match the look's relation count")
    cnt = np.zeros(len(texts), dtype=np.int64)
    for i in rows:
        q = Q[i]
        r = q["rel"][q["fam"] == 0]
        cnt += np.bincount(r[r >= 0], minlength=len(texts))
    order = np.lexsort((np.arange(len(texts)), -cnt))
    rank_of = np.empty(len(texts), dtype=np.int64)
    rank_of[order] = np.arange(len(texts))
    vocab = [texts[k] for k in order]
    phi = np.zeros((R_TOP, 1536), dtype=np.float32)
    k = min(len(texts), R_TOP)
    phi[:k] = emb[order[:k]]
    return rank_of, vocab, phi, [[texts[j], int(cnt[j])] for j in order]


def kb_view(Q, rank_of, one=False):
    """Each row with its structural edges expanded to one walk edge per stored relation (one=True: its most frequent
    only; a pair with none: one edge, 'other'), and w2_f / w2_b laid out as anchor_walk.anchor_tables lays them (over the
    structural edges): the relation's rank in each stored direction, -1 where that direction is not stored."""
    out = [None] * len(Q)
    st = {"structural_edges": 0, "walk_edges": 0, "multi_relation_edges": 0, "both_direction_edges": 0, "no_relation_edges": 0}
    for i, q in enumerate(Q):
        m = q["fam"] == 0
        R = q["rel"]
        has = R >= 0
        nrel = has.sum(1)
        if (nrel[~m] > 0).any():
            raise SystemExit("a non-structural edge carries a relation")
        rk = np.sort(np.where(has, rank_of[np.maximum(R, 0)], R_TOP), axis=1)   # ranks ascending; empty slots last
        rep = np.ones(R.shape[0], dtype=np.int64) if one else np.where(m, np.maximum(nrel, 1), 1)
        idx = np.repeat(np.arange(R.shape[0]), rep)
        j = np.arange(idx.size) - np.repeat(np.cumsum(rep) - rep, rep)
        r_e = rk[idx, j]
        nq = {k: v for k, v in q.items() if k != "rel"}
        for k in ("u", "v", "fam", "fwd", "bwd", "w"):
            nq[k] = q[k][idx]
        sm = nq["fam"] == 0
        nq["w2_f"] = np.where(nq["fwd"][sm], r_e[sm], -1).astype(np.int32)
        nq["w2_b"] = np.where(nq["bwd"][sm], r_e[sm], -1).astype(np.int32)
        out[i] = nq
        st["structural_edges"] += int(m.sum())
        st["walk_edges"] += int(sm.sum())
        st["multi_relation_edges"] += int((m & (nrel > 1)).sum())
        st["both_direction_edges"] += int((m & q["fwd"] & q["bwd"]).sum())
        st["no_relation_edges"] += int((m & (nrel == 0)).sum())
    return out, st


def shuffled(rm, n_rel, K):
    """rm with each of the KB's first n_rel ranks given the next rank's target (a cyclic derangement of the relations)."""
    out = rm.copy()
    n = min(n_rel, R_TOP)
    out[:n] = rm[(np.arange(n) + 1) % n]
    return out


def run(a):
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    paths = [Path(p) for p in a.models.split(",")]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    max_len = max(ck["spec"]["max_len"] for ck in cks)
    Q0, rec_sha, rel, load_state = load_look(a.target, max_len, t0)
    x1 = list(range(len(Q0)))
    B_rows = x1[1::2]
    rank_of, vocab, phi, counts = kb_relations(a.target, Q0, x1, int(rel["n_relations"]))
    n_rel = len(vocab)
    Q, st = kb_view(Q0, rank_of)
    Q1, st1 = kb_view(Q0, rank_of, one=True)
    del Q0
    phi_s = phi.copy()
    k = min(n_rel, R_TOP)
    phi_s[:k] = phi[(np.arange(k) + 1) % k]
    RB, RX = AG.Reader(Q, B_rows, 20261002), AG.Reader(Q, x1, 20261003)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "anchor_gen_kb", "mode": "xread", "target": a.target, "script_sha256": AW.sha(Path(__file__)),
           "pins": {"anchor_gen3": ANCHOR_GEN3_SHA, "anchor_gen": G3.ANCHOR_GEN_SHA, "anchor_gen2": G3.ANCHOR_GEN2_SHA,
                    "look_score_kb_records": rec_sha, "rel_vocab": RELS[a.target][0], "rel_embeddings": RELS[a.target][1]},
           "relations": {**rel, "ranked": counts}, "kb_view": st, "kb_view_1r": st1, "pruned_loader": load_state,
           "B": RB.base(), "x1": RX.base(), "models": {}}
    per_row = {}
    for jm, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp, K = ck["spec"], ck["spec"]["K"]
        # anchor_gen3's loader for its own checkpoints (they record 'pca'), anchor_gen's for anchor_gen's gen models
        model, _ck = (G3.load_model if "pca" in sp else AG.load_model)(p, phi if sp["family"] != "free" else None)
        if sp["family"] != "free":
            G3.set_phi(model, phi, K)
        ent = {"from": ck["dataset"], "name": ck.get("name"), "seed": ck.get("seed"), "hold": ck.get("hold"), "rdrop": ck.get("rdrop"),
               "spec": sp, "margin": ck["margin"], "model_file": str(p)}
        if sp["tok"] != "A":
            nt, TY = AG.types_for(Q, sp, None, x1)
            mB = AG.read(model, Q, TY, nt, B_rows, z_of, ck["margin"])
            mX = AG.read(model, Q, TY, nt, x1, z_of, ck["margin"])
            per_row[f"{jm}_XD"] = mB
            ent["XD"], ent["XD_x1"] = RB.record(mB), RX.record(mX, by_type=False)
        else:
            free = sp["family"] == "free"
            rm = AG.string_map(K, vocab, ck["vocab_w2"]) if free else AG.identity_map(K)
            if free:
                ent["string_matched"] = {vocab[r]: int(rm[r]) for r in range(min(n_rel, R_TOP)) if rm[r] < K}
                ne = nm_ = 0
                for i in B_rows:
                    if (Q[i]["fam"] == 0).any():
                        _m, _d, a_ = AG.edge_ranks(Q[i])
                        ne += int(a_.size)
                        nm_ += int((rm[np.minimum(a_, R_TOP)] < K).sum())
                ent["string_matched_edge_share_B"] = round(nm_ / max(ne, 1), 4)
            nt, TY = AG.types_for(Q, sp, rm, x1)
            _nt, TY1 = AG.types_for(Q1, sp, rm, B_rows)
            _nt, TYn = AG.types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B_rows)
            mB = AG.read(model, Q, TY, nt, B_rows, z_of, ck["margin"])
            mX = AG.read(model, Q, TY, nt, x1, z_of, ck["margin"])
            m1 = AG.read(model, Q1, TY1, nt, B_rows, z_of, ck["margin"])
            mN = AG.read(model, Q, TYn, nt, B_rows, z_of, ck["margin"])
            if free:
                _nt, TYs = AG.types_for(Q, sp, shuffled(rm, n_rel, K), B_rows)
                mS = AG.read(model, Q, TYs, nt, B_rows, z_of, ck["margin"])
            else:
                G3.set_phi(model, phi_s, K)
                mS = AG.read(model, Q, TY, nt, B_rows, z_of, ck["margin"])
                G3.set_phi(model, phi, K)
            per_row.update({f"{jm}_XD": mB, f"{jm}_XD1R": m1, f"{jm}_XDSHUF": mS, f"{jm}_XDNR": mN})
            ent["XD"], ent["XD_x1"] = RB.record(mB), RX.record(mX, by_type=False)
            ent["XD_1R"], ent["XD_SHUF"], ent["XD_NR"] = RB.record(m1, by_type=False), RB.record(mS, by_type=False), RB.record(mN)
            for nm, mm in (("XD_1R", m1), ("XD_SHUF", mS), ("XD_NR", mN)):
                ent[f"{nm} - XD"] = AW3.boot_pair(mm - mB, RB.W)
        ent["seconds"] = round(time.time() - t1, 1)
        res["models"][p.name if p.name not in res["models"] else f"{p.parent.name}/{p.name}"] = ent
        log(f"{p.name} ({ck['dataset']} -> {a.target}, {sp['model']}): XD {ent['XD']['rho (R@5, FC@5, hit@1)']}"
            + (f", XD_NR {ent['XD_NR']['rho (R@5, FC@5, hit@1)']}, XD_SHUF {ent['XD_SHUF']['rho (R@5, FC@5, hit@1)']}" if "XD_NR" in ent else ""))
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="metaqa", choices=tuple(RELS))
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if AW.sha(Path(G3.__file__)) != ANCHOR_GEN3_SHA:
        raise SystemExit("anchor_gen3.py is not the pinned file")
    if AW.sha(Path(AG.__file__)) != G3.ANCHOR_GEN_SHA or AW.sha(Path(G3.G2N.__file__)) != G3.ANCHOR_GEN2_SHA:
        raise SystemExit("anchor_gen.py or anchor_gen2.py is not the pinned file")
    AG.check_pins()
    run(a)


if __name__ == "__main__":
    main()
