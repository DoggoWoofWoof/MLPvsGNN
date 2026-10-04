"""Design look (untracked; not a result and not filed): nearest-seen-phrase backoff, so a relation model reads a relation
it never saw through the closest one it did.

anchor_gen's hold reads show that a phrase the fit never saw gains nothing from its own text code: REV - MASK is +0.000 on
2wiki and hotpot, and on the rows where a hidden phrase reaches a gold REV reads below MASK. A model's codes are trained
for the phrases it saw (a free vector each, or a text map fitted on them). This look keeps every trained code as it is and
changes only which code an edge gets: an edge whose phrase (or, on a KB, whose relation) the model never saw takes the
token of the seen phrase whose gte-Qwen2 vector is closest (cosine c) if c >= theta, else 'other':
    NN@theta   every unseen phrase backs off: in-domain the ranks at or above K and, with a hold, the held fold; on another
               dataset or a KB every phrase its string does not match
    NNH@theta  (in-domain, with a hold) the held fold only, the ranks at or above K kept 'other': unseen relations alone
    BASE       the model's own read (ID; MASK with a hold; on another dataset XD: free by string, text and gen by their own
               text codes); REV (text and gen with a hold: the held fold by its own text codes); NR (every phrase 'other')
A seen phrase keeps its token exactly (in-domain its rank, elsewhere its string's). A text or gen model reads a backed-off
edge with the seen phrase's code from its own training table, so no model reads a code it did not train. theta is chosen
on the source dataset's select carve (--select, in-domain runs only; the mean of R@5 and FC@5) and carried unchanged to
other datasets: a cross-dataset run reports every theta and the summary reads the in-domain choice. The similarity search
uses the 4096-row phrase tables (a KB's relation table) on both sides.

    python outputs/mp_approx_2wiki_anchor/host/anchor_nn.py --target 2wiki --models a.pt,b.pt --out PATH [--select]
    python outputs/mp_approx_2wiki_anchor/host/anchor_nn.py --target metaqa --models a.pt,b.pt --out PATH
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
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_kb_anchor" / "host"))
import anchor_gen3 as G3  # noqa: E402
import anchor_gen_kb as KB  # noqa: E402

AG = G3.AG
AW7, AW6, A16, AW, AW3 = AG.AW7, AG.AW6, AG.A16, AG.AW, AG.AW3
R_TOP = AG.R_TOP
ANCHOR_GEN3_SHA = "ee5f76dc7d391649e04ab2b3575d4fdd30dd62cd2a34b1b7dbd114d56f7b14a7"
ANCHOR_GEN_KB_SHA = "4e09d59de0a60767eb591863352389ff99e164f0e2cd841b858debd309ee7952"
THETAS = (0.95, 0.9, 0.85, 0.8, 0.7, 0.6)
log = AG.log


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def train_table(ck):
    """The phrase table the model trained on (anchor_gen's 1024 rows or anchor_gen3's 4096), by its recorded sha."""
    ds, sha = ck["dataset"], ck.get("phi_sha")
    if ck["spec"]["family"] == "free":
        return None, None
    if sha == AW7.PHRASE[ds][1]:
        return AG.phrase_table(ds)[0], 1024
    if sha == G3.PHRASE4096[ds][1]:
        return G3.phrase_table(ds)[0], 4096
    raise SystemExit(f"{ck.get('name')}: unknown training phrase table {sha}")


def load_any(p, phi):
    """anchor_gen3's loader for its own checkpoints (they record 'pca'), anchor_gen's for anchor_gen's."""
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return G3.load_model(p, phi) if "pca" in ck["spec"] else AG.load_model(p, phi)


def held_of(ck):
    K = ck["spec"]["K"]
    if not ck.get("hold"):
        return np.zeros(R_TOP + 1, dtype=bool)
    h, F = (int(x) for x in ck["hold"].split("/"))
    return AG.hold_map(K, ck["vocab_w2"], h, F)[1]


def exact_map(K, vocab_t, vocab_s, held):
    """Each target rank -> the source rank of the same string if that phrase is seen (in the top K, not held), else K."""
    idx = {s: r for r, s in enumerate(vocab_s[:K]) if not held[r]}
    rm = np.full(R_TOP + 1, K, dtype=np.int64)
    for r, s in enumerate(vocab_t[:R_TOP]):
        rm[r] = idx.get(s, K)
    return rm


def nearest(phi_t, phi_s, seen):
    """Per target row: the source rank of the most similar seen phrase and its cosine."""
    S = unit_rows(phi_s[seen])
    T = unit_rows(phi_t)
    sims = T @ S.T
    j = sims.argmax(1)
    return seen[j].astype(np.int64), sims[np.arange(j.size), j].astype(np.float64)


def backoff(exact, nn_rank, cos, theta, K, scope=None):
    """exact with every unmatched target rank (in scope) sent to its nearest seen phrase when the cosine is at least theta."""
    rm = exact.copy()
    n = nn_rank.size
    back = (exact[:n] >= K) & (cos >= theta)
    if scope is not None:
        back &= scope[:n]
    rm[np.flatnonzero(back)] = nn_rank[back]
    return rm


def typed_share(Q, rows, rm, K):
    ne = nk = 0
    for i in rows:
        if (Q[i]["fam"] == 0).any():
            _m, _d, a_ = AG.edge_ranks(Q[i])
            ne += int(a_.size)
            nk += int((rm[np.minimum(a_, R_TOP)] < K).sum())
    return round(nk / max(ne, 1), 4)


def load_target(target, max_len, select, t0):
    """Rows, carves, the target's phrase strings and its 4096-row (KB: relation) table for the similarity search, and
    a function giving a text model's own-text table for the target (anchor_gen's 1024 or anchor_gen3's 4096 rows)."""
    if target in KB.RELS:
        Q0, rec_sha, rel, load_state = KB.load_look(target, max_len, t0)
        x1 = list(range(len(Q0)))
        rank_of, vocab, phi, counts = KB.kb_relations(target, Q0, x1, int(rel["n_relations"]))
        Q, st = KB.kb_view(Q0, rank_of)
        info = {"kind": "kb", "records": rec_sha, "relations": {**rel, "ranked": counts}, "kb_view": st, "pruned_loader": load_state}
        return Q, {"x1": x1}, vocab, phi, (lambda kind: phi), info
    AW6.rebind(target)
    Q, part, rec_sha, load_state = AG.load_looks(["x1"] + (["select"] if select else []), max_len, t0)
    checks, vocab = AW.anchor_tables(Q)
    phi4096 = G3.phrase_table(target)[0]
    info = {"kind": "passage", "records": rec_sha, "flag_checks": checks, "pruned_loader": load_state,
            "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA}
    return Q, part, vocab.get("w2", []), phi4096, (lambda kind: AG.phrase_table(target)[0] if kind == 1024 else phi4096), info


def run(a):
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    thetas = tuple(float(x) for x in a.thetas.split(","))
    paths = [Path(p) for p in a.models.split(",")]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    if a.select and any(ck["dataset"] != a.target for ck in cks):
        raise SystemExit("--select is for in-domain runs: every model fitted on the target")
    max_len = max(ck["spec"]["max_len"] for ck in cks)
    Q, part, vocab_t, phi_t, own_table, info = load_target(a.target, max_len, a.select, t0)
    x1 = part["x1"]
    B_rows = x1[1::2]
    sel = part.get("select") if a.select else None
    RB = AG.Reader(Q, B_rows, 20261002)
    RS = AG.Reader(Q, sel, 20261004) if sel else None
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "anchor_nn", "target": a.target, "script_sha256": AW.sha(Path(__file__)), "thetas": list(thetas),
           "pins": {"anchor_gen3": ANCHOR_GEN3_SHA, "anchor_gen": G3.ANCHOR_GEN_SHA, "anchor_gen2": G3.ANCHOR_GEN2_SHA,
                    "anchor_gen_kb": AW.sha(Path(KB.__file__))},
           "target_info": info, "B": RB.base(), "select": RS.base() if RS else None, "models": {}}
    per_row = {}
    src_tables = {}
    for jm, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp, K, src = ck["spec"], ck["spec"]["K"], ck["dataset"]
        in_domain = src == a.target
        phi_tr, kind = train_table(ck)
        model, _ck = load_any(p, phi_tr)
        ent = {"from": src, "name": ck.get("name"), "seed": ck.get("seed"), "hold": ck.get("hold"), "rdrop": ck.get("rdrop"), "spec": sp,
               "margin": ck["margin"], "model_file": str(p), "in_domain": in_domain, "reads": {}}
        if sp["tok"] != "A":
            raise SystemExit(f"{p}: anchor_nn reads phrase-token models only")
        held = held_of(ck)
        seen = np.asarray([r for r in range(K) if not held[r]], dtype=np.int64)
        if src not in src_tables:
            src_tables[src] = G3.phrase_table(src)[0]
        nT = min(len(vocab_t), R_TOP)
        exact = exact_map(K, vocab_t, ck["vocab_w2"], held)
        nn_rank, cos = nearest(phi_t[:nT], src_tables[src][:K], seen)
        ent["exact_matched_target_ranks"] = int((exact[:nT] < K).sum())
        ent["nn_cos_quantiles_unmatched"] = (np.quantile(cos[exact[:nT] >= K], [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist()
                                             if (exact[:nT] >= K).any() else None)
        if a.target in KB.RELS:
            ent["kb_nearest"] = {vocab_t[r]: [ck["vocab_w2"][int(nn_rank[r])], round(float(cos[r]), 3), bool(exact[r] < K)] for r in range(nT)}
        maps = {}
        if in_domain:
            maps["BASE"] = exact            # ID, or MASK with a hold: the fit's own map
        elif sp["family"] == "free":
            maps["BASE"] = exact            # XD: string matches
        maps["NR"] = np.full(R_TOP + 1, K, dtype=np.int64)
        for th in thetas:
            maps[f"NN@{th}"] = backoff(exact, nn_rank, cos, th, K)
            if in_domain and held.any():
                maps[f"NNH@{th}"] = backoff(exact, nn_rank, cos, th, K, scope=held)
        own = {}                            # reads with the target's own text codes (text and gen only)
        if sp["family"] != "free":
            if not in_domain:
                own["BASE"] = AG.identity_map(K)
            elif held.any():
                own["REV"] = AG.identity_map(K)

        def read_all(rows, tag):
            out = {}
            for nm, rm in list(maps.items()) + list(own.items()):
                if sp["family"] != "free":
                    G3.set_phi(model, own_table(kind) if nm in own else phi_tr, K)
                _nt, TY = AG.types_for(Q, sp, rm, rows)
                out[nm] = AG.read(model, Q, TY, ck["nt"], rows, z_of, ck["margin"])
                per_row[f"{jm}_{tag}_{nm}"] = out[nm]
            return out

        mB = read_all(B_rows, "B")
        base = mB["BASE"]
        hrows = AG.held_reach_rows(Q, B_rows, held, K) if (in_domain and held.any()) else []
        pos = {i: j for j, i in enumerate(B_rows)}
        hx = np.asarray([pos[i] for i in hrows], dtype=np.int64)
        for nm, m in mB.items():
            e = RB.record(m, by_type=False)
            rm = maps.get(nm, own.get(nm))
            e["typed_edge_share_B"] = typed_share(Q, B_rows, rm, K)
            if nm != "BASE":
                e["minus_BASE"] = AW3.boot_pair(m - base, RB.W)
            if hx.size:
                e["held_reach"] = RB.subset(m, hrows)
                if nm != "BASE":
                    e["held_reach_minus_BASE"] = AW3.boot_pair((m - base)[hx], RB.W[:, hx])
            ent["reads"][nm] = e
        if sel:
            mS = read_all(sel, "S")
            ent["select"] = {nm: m.mean(0).round(5).tolist() for nm, m in mS.items()}
            for fam_ in ("NN", "NNH"):
                cand = [(float(np.mean(mS[f"{fam_}@{th}"].mean(0)[:2])), th) for th in thetas if f"{fam_}@{th}" in mS]
                if cand:
                    best = max(cand, key=lambda x: (x[0], x[1]))
                    ent[f"theta_star_{fam_}"] = best[1]
                    ent[f"select_{fam_}_star_minus_BASE"] = round(best[0] - float(np.mean(mS["BASE"].mean(0)[:2])), 5)
        ent["seconds"] = round(time.time() - t1, 1)
        res["models"][p.name if p.name not in res["models"] else f"{p.parent.name}/{p.name}"] = ent
        log(f"{p.name} ({src} -> {a.target}, {sp['model']}): " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in ent["reads"].items()))
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=AG.DATASETS + tuple(KB.RELS))
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thetas", default=",".join(str(t) for t in THETAS))
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args(argv)
    if AW.sha(Path(G3.__file__)) != ANCHOR_GEN3_SHA:
        raise SystemExit("anchor_gen3.py is not the pinned file")
    if AW.sha(Path(AG.__file__)) != G3.ANCHOR_GEN_SHA or AW.sha(Path(G3.G2N.__file__)) != G3.ANCHOR_GEN2_SHA:
        raise SystemExit("anchor_gen.py or anchor_gen2.py is not the pinned file")
    if ANCHOR_GEN_KB_SHA and AW.sha(Path(KB.__file__)) != ANCHOR_GEN_KB_SHA:
        raise SystemExit("anchor_gen_kb.py is not the pinned file")
    AG.check_pins()
    run(a)


if __name__ == "__main__":
    main()
