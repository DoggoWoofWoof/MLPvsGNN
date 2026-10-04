"""Design look (untracked; not a result and not filed): anchor-typed walks with a frontier-pruned look loader, for musique.

musique's look rows are dense: about 2,080 pool nodes and 60,600 directed pool edges a row, 4,609 rows in a 419-chunk carve.
l16_look_analyze.load's in-memory form (int64 edges, float32 projections) costs about 3 MB a row, so x1, select and x2 alone
would need about 35 GB. The walk types every anchor-walk round reads (l16_look_analyze.walk_types) use only the edges out of a
row's seeds (walks of one edge) and out of the nodes those edges reach (walks of two), and the projections are read only by the
node-gate models (l16_look_gate.pack_gate packs them for every model; the others never read them). This look loads each chunk
as l16_look_analyze.load does, then keeps per row only:
    the edges whose source is a seed (every variant of one edge) or a seed or a node one edge from a seed (any of two edges),
    the projections only when a gate model is asked for, else an (n, 0) array (an empty feature axis in pack_gate).
Every other field is load's, unchanged. So walk_types, and every fit and read, is what the unpruned rows give: on the first
chunk of every look the check below loads the full rows with load's own loop, recomputes each row's walk types from the full
and from the pruned edges (under T0's tokens and under a random token per edge) and stops on any difference. Everything else
is anchor_walk10.py's main (anchor_walk8's loss with or without the GNN teacher, the gates, the rules, the selection, the
reads), whose record this look extends with the pruning counts.

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk11.py --dataset musique --variants x2:A256-1 --kd 0 [--out PATH]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_walk10 as AW10  # noqa: E402

AW8 = AW10.AW8
AW7, AW6, A16, AW = AW8.AW7, AW8.AW6, AW8.A16, AW8.AW
AW10_SHA = "e8efaffde01e158a07d94562275ba6e43281d0b826e03cba6dc572403f3a5cf3"
_load = A16.load
STATE = {"max_len": 2, "proj": True, "rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0}


def keep_mask(u, v, seeds, n, max_len):
    """The edges whose source is a seed, or (max_len 2) a seed or a node one edge from a seed."""
    src = np.zeros(n, dtype=bool)
    S = seeds[seeds >= 0]
    src[S] = True
    if max_len >= 2:
        reach = np.zeros(n, dtype=bool)
        reach[v[src[u]]] = True
        src |= reach
    return src[u]


def same_types(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def check_chunk(path, max_len):
    """load's loop on one chunk, full rows; the walk types of each row from all its edges and from the kept ones."""
    with np.load(path) as zf:
        z = {key: zf[key] for key in zf.files}
    npool, ned = z["q_pool_size"], z["q_edges"]
    po, eo = np.r_[0, np.cumsum(npool)], np.r_[0, np.cumsum(ned)]
    rng = np.random.default_rng(0)
    for i in range(z["chunk_rows"].size):
        a, b, c, d = po[i], po[i + 1], eo[i], eo[i + 1]
        q = {"n": int(npool[i]), "seeds": z["q_seed_local"][i].copy(), "bucket": z["q_seed_bucket"][i].copy(),
             "u": z["e_u"][c:d].astype(np.int64), "v": z["e_v"][c:d].astype(np.int64), "fam": z["e_fam"][c:d].astype(np.int64),
             "fwd": z["e_fwd"][c:d].astype(bool), "bwd": z["e_bwd"][c:d].astype(bool)}
        m = keep_mask(q["u"], q["v"], q["seeds"], q["n"], max_len)
        qp = {**q, **{k: q[k][m] for k in ("u", "v", "fam", "fwd", "bwd")}}
        for tok_full, nt in ((A16.famdir(q), 5), (rng.integers(0, 40, q["u"].size), 40)):
            for L in range(1, max_len + 1):
                if not same_types(A16.walk_types(q, tok_full, nt, L), A16.walk_types(qp, tok_full[m], nt, L)):
                    raise SystemExit(f"{path}: row {i}: pruned walk types differ (max_len {L})")
        STATE["checked_rows"] += 1


def load_pruned(look: Path):
    """l16_look_analyze.load, with each row's edges cut to the walk frontier and its projections dropped unless asked for."""
    look = Path(look)
    ids = json.loads((look / "ids.json").read_text(encoding="utf-8"))
    info = json.loads((look / "info.json").read_text(encoding="utf-8"))
    Q = [None] * len(ids)
    paths = sorted((look / "chunks").glob("c*.npz"))
    all_rows = np.sort(np.concatenate([np.load(p)["chunk_rows"] for p in paths]))
    if all_rows.size != len(ids) or np.unique(all_rows).size != all_rows.size:
        raise SystemExit("the chunks do not hold each scored row once")
    check_chunk(paths[0], STATE["max_len"])
    for p in paths:
        with np.load(p) as zf:
            z = {key: zf[key] for key in zf.files}
        npool, ned = z["q_pool_size"], z["q_edges"]
        po, eo = np.r_[0, np.cumsum(npool)], np.r_[0, np.cumsum(ned)]
        for i, row in enumerate(z["chunk_rows"]):
            a, b = po[i], po[i + 1]
            c, d = eo[i], eo[i + 1]
            n = int(npool[i])
            u, v = z["e_u"][c:d].astype(np.int64), z["e_v"][c:d].astype(np.int64)
            seeds = z["q_seed_local"][i].copy()
            m = keep_mask(u, v, seeds, n, STATE["max_len"])
            STATE["rows"] += 1
            STATE["edges_in"] += int(u.size)
            STATE["edges_kept"] += int(m.sum())
            proj = z["proj"][a:b].astype(np.float32) if STATE["proj"] else np.zeros((n, 0), dtype=np.float32)
            Q[int(np.searchsorted(all_rows, row))] = {
                "n": n, "gt": int(z["q_gold_total"][i]), "metrics": z["q_metrics"][i].copy(), "qemb": z["q_emb"][i].copy(),
                "seeds": seeds, "bucket": z["q_seed_bucket"][i].copy(), "pool": z["pool"][a:b].copy(),
                "score": z["score"][a:b].copy(), "gold": z["is_gold"][a:b].copy(), "proj": proj,
                "u": u[m], "v": v[m], "fam": z["e_fam"][c:d][m].astype(np.int64),
                "fwd": z["e_fwd"][c:d][m].astype(bool), "bwd": z["e_bwd"][c:d][m].astype(bool), "w": z["e_w"][c:d][m].copy()}
        del z
    missing = [i for i, q in enumerate(Q) if q is None]
    if missing:
        raise SystemExit(f"{len(missing)} rows missing from the chunks")
    for q, inf in zip(Q, info):
        q["type"] = inf["type"]
    return ids, Q


def needs(argv):
    """(max_len, gate) over the variants asked for."""
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "2wiki"
    vs = argv[argv.index("--variants") + 1].split(",")
    specs = [AW10.parse(v, AW6.TRAIN[ds]) for v in vs]
    return max(s["max_len"] for s in specs), any(s["model"] in AW10.GATES for s in specs)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if AW.sha(Path(AW10.__file__)) != AW10_SHA:
        raise SystemExit("anchor_walk10.py is not the pinned file")
    if AW.sha(Path(A16.__file__)) != AW.A16_SHA:
        raise SystemExit("l16_look_analyze.py is not the pinned file")
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "2wiki"
    if "--out" not in argv:
        argv += ["--out", str(HERE / f"anchor_walk11_{ds}.json")]
    out = Path(argv[argv.index("--out") + 1])
    STATE["max_len"], STATE["proj"] = needs(argv)
    STATE.update({"rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0})
    t0 = time.time()
    A16.load = load_pruned
    try:
        AW10.main(argv)
    finally:
        A16.load = _load
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "anchor_walk11"
    res["pins"]["anchor_walk10"] = AW10_SHA
    res["pins"]["anchor_walk10_record_sha256_field"] = res["script_sha256"]
    res["script_sha256"] = AW.sha(Path(__file__))
    res["pruned_loader"] = {"max_len": STATE["max_len"], "projections_kept": STATE["proj"], "rows": STATE["rows"], "edges_in": STATE["edges_in"],
                            "edges_kept": STATE["edges_kept"], "walk_type_check_rows": STATE["checked_rows"], "seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
