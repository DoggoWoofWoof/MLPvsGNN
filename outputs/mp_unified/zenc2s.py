"""ENC E2S (docs/ENC_ENCODER_COMPARISON.md section 14, declared 20 Nov 2026 06:40 IST before any E2S number): can mpnet's
vectors be improved without touching mpnet? Three arms, read at the first stage on E0's s1eval carves (SQuAD, MuSiQue),
each arm's setting chosen on the s1sel carves.

    python outputs/mp_unified/zenc2s.py selftest [--real]     (--real: on the host, the real carves, pairs and windows)
    python outputs/mp_unified/zenc2s.py fuse --threads 8      (host CPU)
    python outputs/mp_unified/zenc2s.py adapt --threads 8     (host CPU; HotpotQA's unseen read included)
    python outputs/mp_unified/zenc2s.py winenc --device cuda  (host GPU: the window vectors, and G2)
    python outputs/mp_unified/zenc2s.py win --threads 8       (host CPU)
    python outputs/mp_unified/zenc2s.py report

Arms (14.2): win = (1 - beta) cos(q, d) + beta max over d's windows; fuse = (1 - w) cos_mpnet + w cos_mpnetqa; adapt = a
learned map f(x) = unit(x + g(x)) on mpnet's question vector, the documents untouched. The call (14.5) is on s1eval dense
R@5 won back against E0's raw mpnet read. Nothing here changes a model's weights, a world or anything outside
outputs/enc/e2s/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)
import zenc2m as M  # noqa: E402

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "enc" / "e2s"
E0 = ROOT / "outputs" / "enc" / "e0"
CARVES = ROOT / "outputs" / "step1" / "carves.json"
STAGE = "ENC E2S"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 14"
DS = ("squad", "musique")
UNSEEN = "hotpotqa"
SEED = 0
WINS = (64, 126)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FORMS = ("linear", "mlp")
LRS = (1e-4, 3e-4, 1e-3)
EPOCHS, BATCH, TAU, NHARD, MINE, WD, HID = 10, 512, 0.05, 16, 50, 0.01, 2048
G1_SHARE, G2_COS, G2_SHARE = 0.995, 0.999, 0.995
QB = 256


def log(msg):
    print(f"[zenc2s {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc2s: {msg}")


def code_shas():
    return {f"outputs/mp_unified/{Path(p).name}": E.sha256(Path(p)) for p in (__file__, E.__file__, M.__file__)}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── questions, golds and carves ──────────────────────────────────────────────

def query_golds(root, ds):
    """Every query row's gold node ids, in the manifest's split order (row j = query_ids.json[j], checked)."""
    man = E.manifest(root, ds, "queries")
    qids = json.loads((root / ds / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    out, j = [], 0
    for sp in E.query_splits(man):
        with open(root / ds / "queries" / f"{sp}.jsonl", encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                if r.get("query_id") != qids[j]:
                    stop(f"{ds} query row {j}: {r.get('query_id')!r} is not query_ids.json's {qids[j]!r}")
                g = r.get("gold_node_ids") or []
                out.append(list(json.loads(g) if isinstance(g, str) else g))
                j += 1
    if j != man["n_rows"] or j != len(qids):
        stop(f"{ds}: {j} query lines, {len(qids)} ids, manifest {man['n_rows']}")
    return out, [str(q) for q in qids]


def node_rows(root, ds, need):
    pos, pat = {}, re.compile(r'"node_id":\s*"([^"]+)"')
    with open(root / ds / "nodes.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            m = pat.search(ln)
            if m and m.group(1) in need:
                pos[m.group(1)] = i
    return pos


def gold_rows(root, ds):
    """(per query row: its golds' document rows, sorted and unique; the query ids). Unresolved gold ids are dropped."""
    g, qids = query_golds(root, ds)
    pos = node_rows(root, ds, {n for x in g for n in x})
    return [np.unique(np.asarray([pos[n] for n in x if n in pos], np.int64)) for x in g], qids


@functools.lru_cache(maxsize=1)
def carves_json():
    return json.loads(CARVES.read_text(encoding="utf-8"))


def ragged(gl):
    return np.concatenate([np.zeros(0, np.int64)] + list(gl)), np.concatenate([[0], np.cumsum([x.size for x in gl])]).astype(np.int64)


def sel_carve(ds, golds, qids):
    """The s1sel carve: carves.json's dselect ids, in order, with their golds."""
    ids = [str(x) for x in carves_json()["per_dataset"][ds]["dselect"]["ids"]]
    if len(ids) != carves_json()["per_dataset"][ds]["dselect"]["n"]:
        stop(f"{ds} s1sel: {len(ids)} ids, carves.json says {carves_json()['per_dataset'][ds]['dselect']['n']}")
    qmap = {q: j for j, q in enumerate(qids)}
    miss = [q for q in ids if q not in qmap]
    if miss:
        stop(f"{ds} s1sel: {len(miss)} ids are not query ids ({miss[:3]})")
    rows = np.asarray([qmap[q] for q in ids], np.int64)
    gl = [golds[r] for r in rows]
    if any(x.size == 0 for x in gl):
        stop(f"{ds} s1sel: a question has no resolved gold")
    flat, off = ragged(gl)
    return {"ids": ids, "rows": rows, "flat": flat, "off": off, "gt": np.diff(off)}


def rows_of(root, enc, ds, kind, rows):
    """float32 vectors of the given rows, shard by shard (memory-mapped)."""
    paths = E.vec_shards(root, enc, ds, kind)
    rows = np.asarray(rows, np.int64)
    out = np.empty((rows.size, E.ENCODERS[enc]["dim"]), np.float32)
    for s in np.unique(rows // E.SHARD):
        A = np.load(paths[int(s)], mmap_mode="r")
        m = rows // E.SHARD == s
        out[m] = A[rows[m] - s * E.SHARD]
    return out


# ── search and reads ─────────────────────────────────────────────────────────

def topk(score_fn, nq, k, qb=QB):
    """Per question, the k highest scores of score_fn(q0, q1) (rows q0:q1 against every document), by score then id."""
    ids = np.empty((nq, k), np.int64)
    sc = np.empty((nq, k), np.float32)
    for q0 in range(0, nq, qb):
        q1 = min(nq, q0 + qb)
        S = score_fn(q0, q1)
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.lexsort((part, -ps), axis=1)
        ids[q0:q1] = np.take_along_axis(part, o, 1)
        sc[q0:q1] = np.take_along_axis(ps, o, 1)
    return ids, sc


def recall_share(h, gt):
    return float(np.mean(np.minimum(h, gt) / np.asarray(gt, np.float64)))


def r5_of(top, c):
    return recall_share(E.hits(top[:, :5], c["flat"], c["off"]), c["gt"])


def read_of(top, c):
    r5 = E.rrf_top(top, c["spl"], 5)
    return {"d5": E.hits(top[:, :5], c["flat"], c["off"]), "d20": E.hits(top[:, :20], c["flat"], c["off"]),
            "d1000": E.hits(top, c["flat"], c["off"]), "rrf5": E.hits(r5, c["flat"], c["off"])}


def save_read(name, ds, c, h, top):
    d = OUT / "reads"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}__{ds}.npz"
    np.savez(p, ids=np.asarray(c["ids"]), rows=c["rows"], gold_total=c["gt"], top20=top[:, :20].astype(np.int32),
             **{f"h_{k}": v.astype(np.int16) for k, v in h.items()})
    return {"npz": p.name, "sha256": E.sha256(p), "recall": {k: round(recall_share(v, c["gt"]), 6) for k, v in h.items()}}


def g1_check(h, enc, ds, c):
    z = np.load(E0 / enc / f"{ds}.npz", allow_pickle=False)
    if [str(x) for x in z["ids"]] != list(c["ids"]):
        stop(f"{ds}: E0's {enc} read holds other questions")
    same = {k: round(float(np.mean(h[k] == z[f"h_{k}"])), 6) for k in h}
    return {"equal_share": same, "pass": all(v >= G1_SHARE for v in same.values())}


def pick(curve, key, order):
    """The entry with the highest key; ties go to the smallest order tuple (the setting nearest raw mpnet)."""
    best = None
    for x in curve:
        if best is None or key(x) > key(best) or (key(x) == key(best) and order(x) < order(best)):
            best = x
    return best


# ── fuse ─────────────────────────────────────────────────────────────────────

def fuse_fn(Q, D, w):
    def f(q0, q1):
        if w == 0.0:
            return Q["mpnet"][q0:q1] @ D["mpnet"].T
        if w == 1.0:
            return Q["mpnetqa"][q0:q1] @ D["mpnetqa"].T
        return (np.float32(1.0 - w) * (Q["mpnet"][q0:q1] @ D["mpnet"].T)
                + np.float32(w) * (Q["mpnetqa"][q0:q1] @ D["mpnetqa"].T))
    return f


def cmd_fuse(a):
    root = E.served_root()
    t0 = time.time()
    if (OUT / "fuse.json").exists():
        stop("outputs/enc/e2s/fuse.json exists: the arm is read once")
    data = {}
    for ds in DS:
        golds, qids = gold_rows(root, ds)
        cs, ce = sel_carve(ds, golds, qids), M.carve(root, ds)
        D = {e: M.load_all(root, e, ds, "docs") for e in ("mpnet", "mpnetqa")}
        data[ds] = (cs, ce, D, {e: rows_of(root, e, ds, "queries", cs["rows"]) for e in D},
                    {e: rows_of(root, e, ds, "queries", ce["rows"]) for e in D})
    curve = []
    for w in WEIGHTS:
        r = {ds: r5_of(topk(fuse_fn(data[ds][3], data[ds][2], w), len(data[ds][0]["ids"]), 5)[0], data[ds][0])
             for ds in DS}
        curve.append({"w": w, "s1sel_r5": {k: round(v, 6) for k, v in r.items()}, "mean": float(np.mean(list(r.values())))})
        log(f"fuse s1sel w={w}: {json.dumps(curve[-1]['s1sel_r5'])}")
    best = pick(curve, key=lambda x: x["mean"], order=lambda x: (x["w"],))
    g1, reads = {"mpnet": {}, "mpnetqa": {}}, {}
    for ds in DS:
        cs, ce, D, _, Qe = data[ds]
        for enc, w in (("mpnet", 0.0), ("mpnetqa", 1.0)):
            top, _ = topk(fuse_fn(Qe, D, w), len(ce["ids"]), E.K)
            g1[enc][ds] = g1_check(read_of(top, ce), enc, ds, ce)
        top, _ = topk(fuse_fn(Qe, D, best["w"]), len(ce["ids"]), E.K)
        reads[f"fuse__{ds}"] = save_read("fuse", ds, ce, read_of(top, ce), top)
        log(f"fuse s1eval {ds} (w={best['w']}): {json.dumps(reads[f'fuse__{ds}']['recall'])}")
    rec = {"stage": STAGE, "arm": "fuse", "declared_in": DECLARED, "curve": curve, "chosen": {"w": best["w"]},
           "g1": g1, "reads": reads, "threads": E.THREADS, "code_sha256": code_shas(),
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    E.wjson(OUT / "fuse.json", rec)
    log(f"fuse done: w={best['w']}, G1 {json.dumps({e: {d: v['pass'] for d, v in g1[e].items()} for e in g1})}")
    return 0


# ── adapt ────────────────────────────────────────────────────────────────────

def build_pairs(root, ds, golds, sel):
    """14.2's pairs: train-split questions, less s1sel's and those sharing a gold with s1sel or with any non-train
    question; one (question, gold) pair per gold."""
    sp = M.split_rows(root, ds)
    tr = sp["train"]
    nontrain = np.setdiff1d(np.arange(len(golds)), tr)
    bad = set(ragged([golds[r] for r in nontrain])[0].tolist())
    selg, selr = set(sel["flat"].tolist()), set(sel["rows"].tolist())
    keep, why = [], {"in_s1sel": 0, "shares_s1sel_gold": 0, "shares_nontrain_gold": 0, "no_gold": 0}
    for r in tr.tolist():
        g = set(golds[r].tolist())
        if r in selr:
            why["in_s1sel"] += 1
        elif not g:
            why["no_gold"] += 1
        elif g & selg:
            why["shares_s1sel_gold"] += 1
        elif g & bad:
            why["shares_nontrain_gold"] += 1
        else:
            keep.append(r)
    keep = np.asarray(keep, np.int64)
    pq = np.concatenate([np.full(golds[r].size, i, np.int64) for i, r in enumerate(keep)])
    pd = np.concatenate([golds[r] for r in keep])
    info = {"train_questions": int(tr.size), "kept_questions": int(keep.size), "pairs": int(pq.size), "dropped": why}
    return keep, pq, pd, bad, info


def hard_negs(top, keep, golds):
    """Per kept question, its first NHARD non-gold documents of the ranked list top (nq, MINE)."""
    out = np.empty((keep.size, NHARD), np.int64)
    for i, r in enumerate(keep.tolist()):
        h = top[i][~np.isin(top[i], golds[r])][:NHARD]
        if h.size < NHARD:
            stop(f"question row {r}: {h.size} non-gold documents in its top {MINE}")
        out[i] = h
    return out


def make_g(form, dim):
    import torch
    if form == "linear":
        g = torch.nn.Linear(dim, dim)
        with torch.no_grad():
            g.weight.zero_()
            g.bias.zero_()
        return g
    up, down = torch.nn.Linear(dim, HID), torch.nn.Linear(HID, dim)
    with torch.no_grad():
        down.weight.zero_()
        down.bias.zero_()
    return torch.nn.Sequential(up, torch.nn.GELU(), down)


def snapshot(g):
    return {k: v.detach().clone().numpy() for k, v in g.state_dict().items()}


def apply_map(state, form, X, block=65536):
    """unit(x + g(x)) for the rows of X (float32), with g loaded from state."""
    import torch
    g = make_g(form, X.shape[1])
    g.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state.items()})
    out = np.empty_like(X, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], block):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i + block], np.float32))
            out[i:i + block] = torch.nn.functional.normalize(x + g(x), dim=1).numpy()
    return out


def train_map(data, form, lr, epochs, sel_fn, score, seed=SEED, batch=BATCH):
    """data[ds] = {Q (nq, dim), D (nd, dim), pq, pd, hard (nq, NHARD), keys (sorted pq * nd + pd)}. After each epoch
    (epoch 0 = raw) sel_fn(state) gives per-dataset s1sel R@5; score(sel) ranks epochs, ties to the earlier epoch.
    Returns (curve, best state, best epoch)."""
    import torch
    F = torch.nn.functional
    torch.manual_seed(seed)
    dim = next(iter(data.values()))["Q"].shape[1]
    g = make_g(form, dim)
    opt = torch.optim.AdamW(g.parameters(), lr=lr, weight_decay=WD)
    T = {ds: {k: torch.from_numpy(v[k]) for k in ("Q", "D", "pq", "pd", "hard", "keys")} for ds, v in data.items()}
    gen = torch.Generator().manual_seed(seed)
    st = snapshot(g)
    sel = sel_fn(st)
    curve = [{"epoch": 0, "sel": {k: round(v, 6) for k, v in sel.items()}, "score": score(sel), "loss": None}]
    best = (score(sel), 0, st)
    for ep in range(1, epochs + 1):
        batches = []
        for ds in T:
            n = T[ds]["pq"].numel()
            perm = torch.randperm(n, generator=gen)
            batches += [(ds, perm[i:i + batch]) for i in range(0, n, batch)]
        losses = []
        for bi in torch.randperm(len(batches), generator=gen).tolist():
            ds, idx = batches[bi]
            t = T[ds]
            qi, di = t["pq"][idx], t["pd"][idx]
            x = t["Q"][qi]
            fx = F.normalize(x + g(x), dim=1)
            lb = fx @ t["D"][di].T
            lh = torch.einsum("bd,bkd->bk", fx, t["D"][t["hard"][qi]])
            nd = t["D"].shape[0]
            m = torch.isin(qi[:, None] * nd + di[None, :], t["keys"])
            m.fill_diagonal_(False)
            logits = torch.cat([lb.masked_fill(m, float("-inf")), lh], 1) / TAU
            loss = F.cross_entropy(logits, torch.arange(idx.numel()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        st = snapshot(g)
        sel = sel_fn(st)
        curve.append({"epoch": ep, "sel": {k: round(v, 6) for k, v in sel.items()}, "score": score(sel),
                      "loss": round(float(np.mean(losses)), 6)})
        if score(sel) > best[0]:
            best = (score(sel), ep, st)
    return curve, best[1], best[2]


def adapt_data(root, enc, ds, golds, sel):
    """The training block for one encoder and dataset: kept questions' vectors, the documents, pairs, hard negatives."""
    keep, pq, pd, bad, info = build_pairs(root, ds, golds, sel)
    Q = rows_of(root, enc, ds, "queries", keep)
    D = M.load_all(root, enc, ds, "docs")
    if enc == "frozen":
        top = np.asarray(np.load(root / ds / "retrieval_cache" / "dense_top1000.npz", mmap_mode="r")["ids"][keep][:, :MINE],
                         np.int64)
    else:
        top, _ = topk(lambda q0, q1: Q[q0:q1] @ D.T, keep.size, MINE, qb=1024)
    hard = hard_negs(top, keep, golds)
    keys = np.unique(pq * D.shape[0] + pd)
    return {"Q": Q, "D": D, "pq": pq, "pd": pd, "hard": hard, "keys": keys}, bad, info


def sel_reader(sels, Ds, form):
    """sel_fn for train_map: the map applied to each dataset's s1sel questions, R@5 over the whole corpus."""
    def f(state):
        out = {}
        for ds, (c, Qs) in sels.items():
            Qm = apply_map(state, form, Qs)
            out[ds] = r5_of(topk(lambda q0, q1: Qm[q0:q1] @ Ds[ds].T, Qm.shape[0], 5)[0], c)
        return out
    return f


def save_map(tag, state, form):
    d = OUT / "maps"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{tag}.npz"
    np.savez(p, form=np.asarray(form), **{k.replace(".", "__"): v for k, v in state.items()})
    return {"npz": p.name, "sha256": E.sha256(p)}


def cmd_adapt(a):
    import torch
    torch.set_num_threads(E.THREADS)
    root = E.served_root()
    t0 = time.time()
    if (OUT / "adapt.json").exists():
        stop("outputs/enc/e2s/adapt.json exists: the arm is read once")
    golds, sels, ces, blocks, info = {}, {}, {}, {}, {}
    for ds in DS:
        golds[ds], qids = gold_rows(root, ds)
        cs = sel_carve(ds, golds[ds], qids)
        ces[ds] = M.carve(root, ds)
        blocks[ds], bad, info[ds] = adapt_data(root, "mpnet", ds, golds[ds], cs)
        if not np.isin(ces[ds]["flat"], np.asarray(sorted(bad), np.int64)).all():
            stop(f"{ds}: an s1eval gold is not a non-train question's gold")
        sels[ds] = (cs, rows_of(root, "mpnet", ds, "queries", cs["rows"]))
        log(f"adapt pairs {ds}: {json.dumps(info[ds])}")
    Ds = {ds: blocks[ds]["D"] for ds in DS}
    runs, best = [], None
    for fi, form in enumerate(FORMS):
        for li, lr in enumerate(LRS):
            t = time.time()
            curve, ep, st = train_map(blocks, form, lr, EPOCHS, sel_reader(sels, Ds, form),
                                      score=lambda s: float(np.mean([s[d] for d in DS])))
            run = {"form": form, "lr": lr, "best_epoch": ep, "score": curve[ep]["score"], "curve": curve,
                   "seconds": round(time.time() - t, 1)}
            runs.append(run)
            log(f"adapt {form} lr={lr}: best epoch {ep}, s1sel mean R@5 {round(curve[ep]['score'], 6)} "
                f"(epoch 0 {round(curve[0]['score'], 6)}), {run['seconds']} s")
            cand = (run["score"], (ep, fi, li), form, lr, ep, st)
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                best = cand
    _, _, form, lr, ep, st = best
    chosen = {"form": form, "lr": lr, "epoch": ep}
    log(f"adapt chosen: {json.dumps(chosen)}")
    maps = {"adapt": save_map("adapt", st, form)}
    g1, reads = {}, {}
    for ds in DS:
        c = ces[ds]
        Q = rows_of(root, "mpnet", ds, "queries", c["rows"])
        paths = E.vec_shards(root, "mpnet", ds, "docs")
        top, _, _ = E.search(Q, paths, E.K)
        g1[ds] = g1_check(read_of(top, c), "mpnet", ds, c)
        top, _, _ = E.search(apply_map(st, form, Q), paths, E.K)
        reads[f"adapt__{ds}"] = save_read("adapt", ds, c, read_of(top, c), top)
        log(f"adapt s1eval {ds}: {json.dumps(reads[f'adapt__{ds}']['recall'])}")
    # beside: cross maps (one dataset's pairs, read on the other), epochs by their own dataset's s1sel
    cross = {}
    for src, dst in (("squad", "musique"), ("musique", "squad")):
        curve, cep, cst = train_map({src: blocks[src]}, form, lr, EPOCHS, sel_reader({src: sels[src]}, Ds, form),
                                    score=lambda s, src=src: s[src])
        maps[f"adapt-cross-{src}"] = save_map(f"adapt-cross-{src}", cst, form)
        c = ces[dst]
        Q = rows_of(root, "mpnet", dst, "queries", c["rows"])
        top, _, _ = E.search(apply_map(cst, form, Q), E.vec_shards(root, "mpnet", dst, "docs"), E.K)
        reads[f"adapt-cross-{src}__{dst}"] = save_read(f"adapt-cross-{src}", dst, c, read_of(top, c), top)
        cross[src] = {"read_on": dst, "best_epoch": cep, "curve": curve}
        log(f"adapt cross {src} -> {dst}: epoch {cep}, {json.dumps(reads[f'adapt-cross-{src}__{dst}']['recall'])}")
    del blocks
    # beside: the same recipe on the frozen vectors
    fb, fsel = {}, {}
    for ds in DS:
        cs = sels[ds][0]
        fb[ds], _, _ = adapt_data(root, "frozen", ds, golds[ds], cs)
        fsel[ds] = (cs, rows_of(root, "frozen", ds, "queries", cs["rows"]))
    curve, fep, fst = train_map(fb, form, lr, EPOCHS, sel_reader(fsel, {ds: fb[ds]["D"] for ds in DS}, form),
                                score=lambda s: float(np.mean([s[d] for d in DS])))
    maps["adapt-frozen"] = save_map("adapt-frozen", fst, form)
    del fb
    for ds in DS:
        c = ces[ds]
        Q = rows_of(root, "frozen", ds, "queries", c["rows"])
        top, _, _ = E.search(apply_map(fst, form, Q), E.vec_shards(root, "frozen", ds, "docs"), E.K)
        reads[f"adapt-frozen__{ds}"] = save_read("adapt-frozen", ds, c, read_of(top, c), top)
        log(f"adapt frozen {ds}: epoch {fep}, {json.dumps(reads[f'adapt-frozen__{ds}']['recall'])}")
    # the unseen read: HotpotQA's s1eval over mpnet's whole index
    c = M.carve(root, UNSEEN)
    Q = rows_of(root, "mpnet", UNSEEN, "queries", c["rows"])
    paths = E.vec_shards(root, "mpnet", UNSEEN, "docs")
    top, _, _ = E.search(Q, paths, E.K)
    g1[UNSEEN] = g1_check(read_of(top, c), "mpnet", UNSEEN, c)
    top, _, _ = E.search(apply_map(st, form, Q), paths, E.K)
    reads[f"adapt__{UNSEEN}"] = save_read("adapt", UNSEEN, c, read_of(top, c), top)
    log(f"adapt unseen {UNSEEN}: {json.dumps(reads[f'adapt__{UNSEEN}']['recall'])}")
    rec = {"stage": STAGE, "arm": "adapt", "declared_in": DECLARED, "pairs": info, "runs": runs, "chosen": chosen,
           "cross": cross, "frozen": {"best_epoch": fep, "curve": curve}, "g1": g1, "reads": reads, "maps": maps,
           "threads": E.THREADS, "code_sha256": code_shas(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    E.wjson(OUT / "adapt.json", rec)
    log(f"adapt done in {rec['seconds']} s; G1 {json.dumps({d: v['pass'] for d, v in g1.items()})}")
    return 0


# ── win ──────────────────────────────────────────────────────────────────────

def window_spans(offsets, W):
    """Token windows of W at stride W // 2 as character spans; the last window ends at the last token. A text of W
    tokens or fewer is one window."""
    L = len(offsets)
    if L == 0:
        return [None]
    if L <= W:
        return [(offsets[0][0], offsets[-1][1])]
    s = W // 2
    n = 1 + -(-(L - W) // s)
    starts = [min(k * s, L - W) for k in range(n)]
    return [(offsets[t][0], offsets[t + W - 1][1]) for t in starts]


def windows_of(tok, texts, W):
    """(window texts, document index per window) for the texts, by window_spans on the tokenizer's offsets."""
    enc = tok(texts, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    out, doc = [], []
    for i, (t, offs) in enumerate(zip(texts, enc["offset_mapping"])):
        for sp in window_spans(offs, W):
            out.append(t if sp is None else t[sp[0]:sp[1]])
            doc.append(i)
    return out, np.asarray(doc, np.int64)


def cmd_winenc(a):
    root = E.served_root()
    t0 = time.time()
    if a.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            stop("no CUDA device")
    d = OUT / "win"
    if (d / "winenc.json").exists():
        stop("outputs/enc/e2s/win/winenc.json exists: the windows are encoded once")
    d.mkdir(parents=True, exist_ok=True)
    m = E.load_model("mpnet", a.device, E.pinned("mpnet"))
    rec = {"stage": STAGE, "arm": "win", "declared_in": DECLARED, "device": a.device, "sets": {}, "g2": {}}
    for ds in DS:
        texts = [t for _, ch in E.doc_chunks(root, ds) for t in ch]
        own = M.load_all(root, "mpnet", ds, "docs")
        if len(texts) != own.shape[0]:
            stop(f"{ds}: {len(texts)} texts, {own.shape[0]} vectors")
        for W in WINS:
            t = time.time()
            wt, doc = windows_of(m.tokenizer, texts, W)
            V, stt = E.encode_texts(m, "mpnet", wt, "docs")
            np.save(d / f"{ds}_w{W}.npy", V)
            np.save(d / f"{ds}_w{W}_doc.npy", doc.astype(np.int32))
            one = np.flatnonzero(np.bincount(doc, minlength=len(texts)) == 1)
            first = np.searchsorted(doc, one)
            cos = np.sum(M.unit(V[first].astype(np.float32)) * M.unit(own[one]), axis=1)
            share = float(np.mean(cos >= G2_COS)) if one.size else 1.0
            rec["g2"][f"{ds}_w{W}"] = {"one_window_docs": int(one.size), "share_ge": round(share, 6),
                                       "min_cos": round(float(cos.min()), 6) if one.size else None,
                                       "pass": share >= G2_SHARE}
            rec["sets"][f"{ds}_w{W}"] = {"windows": int(doc.size), "docs": len(texts), "stats": stt,
                                         "sha256": E.sha256(d / f"{ds}_w{W}.npy"),
                                         "doc_sha256": E.sha256(d / f"{ds}_w{W}_doc.npy"),
                                         "seconds": round(time.time() - t, 1)}
            log(f"winenc {ds} W={W}: {doc.size} windows, G2 {json.dumps(rec['g2'][f'{ds}_w{W}'])}")
    rec.update(code_sha256=code_shas(), seconds=round(time.time() - t0, 1), utc=utc())
    E.wjson(d / "winenc.json", rec)
    return 0


def win_fn(Q, D, Wv, starts, beta):
    def f(q0, q1):
        if beta == 0.0:
            return Q[q0:q1] @ D.T
        mx = np.maximum.reduceat(Q[q0:q1] @ Wv.T, starts, axis=1)
        if beta == 1.0:
            return mx
        return np.float32(1.0 - beta) * (Q[q0:q1] @ D.T) + np.float32(beta) * mx
    return f


def load_windows(ds, W, nd):
    d = OUT / "win"
    Wv = np.asarray(np.load(d / f"{ds}_w{W}.npy"), np.float32)
    doc = np.load(d / f"{ds}_w{W}_doc.npy").astype(np.int64)
    if doc.size != Wv.shape[0] or np.any(np.diff(doc) < 0):
        stop(f"{ds} W={W}: the window index is not sorted by document")
    starts = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
    if starts.size != nd or not np.array_equal(doc[starts], np.arange(nd)):
        stop(f"{ds} W={W}: not every document has a window")
    return Wv, starts


def cmd_win(a):
    root = E.served_root()
    t0 = time.time()
    if (OUT / "win.json").exists():
        stop("outputs/enc/e2s/win.json exists: the arm is read once")
    enc_rec = json.loads((OUT / "win" / "winenc.json").read_text(encoding="utf-8"))
    if not all(v["pass"] for v in enc_rec["g2"].values()):
        stop(f"G2 failed: {json.dumps(enc_rec['g2'])}")
    data = {}
    for ds in DS:
        golds, qids = gold_rows(root, ds)
        cs, ce = sel_carve(ds, golds, qids), M.carve(root, ds)
        D = M.load_all(root, "mpnet", ds, "docs")
        data[ds] = (cs, ce, D, rows_of(root, "mpnet", ds, "queries", cs["rows"]),
                    rows_of(root, "mpnet", ds, "queries", ce["rows"]),
                    {W: load_windows(ds, W, D.shape[0]) for W in WINS})
    curve = []
    for W in WINS:
        for beta in BETAS:
            r = {}
            for ds in DS:
                cs, _, D, Qs, _, wins = data[ds]
                r[ds] = r5_of(topk(win_fn(Qs, D, *wins[W], beta), len(cs["ids"]), 5)[0], cs)
            curve.append({"W": W, "beta": beta, "s1sel_r5": {k: round(v, 6) for k, v in r.items()},
                          "mean": float(np.mean(list(r.values())))})
            log(f"win s1sel W={W} beta={beta}: {json.dumps(curve[-1]['s1sel_r5'])}")
    best = pick(curve, key=lambda x: x["mean"], order=lambda x: (x["beta"], x["W"]))
    g1, reads = {}, {}
    for ds in DS:
        cs, ce, D, _, Qe, wins = data[ds]
        top, _ = topk(win_fn(Qe, D, None, None, 0.0), len(ce["ids"]), E.K)
        g1[ds] = g1_check(read_of(top, ce), "mpnet", ds, ce)
        top, _ = topk(win_fn(Qe, D, *wins[best["W"]], best["beta"]), len(ce["ids"]), E.K)
        reads[f"win__{ds}"] = save_read("win", ds, ce, read_of(top, ce), top)
        log(f"win s1eval {ds} (W={best['W']}, beta={best['beta']}): {json.dumps(reads[f'win__{ds}']['recall'])}")
    rec = {"stage": STAGE, "arm": "win", "declared_in": DECLARED, "curve": curve,
           "chosen": {"W": best["W"], "beta": best["beta"]}, "g1": g1, "g2": enc_rec["g2"], "reads": reads,
           "threads": E.THREADS, "code_sha256": code_shas(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    E.wjson(OUT / "win.json", rec)
    log(f"win done: {json.dumps(rec['chosen'])}, G1 {json.dumps({d: v['pass'] for d, v in g1.items()})}")
    return 0


# ── the report ───────────────────────────────────────────────────────────────

def call_of(wb):
    """14.5 on the won-back cells {ds: gap dict}: IMPROVES, MIXED or NO."""
    lo = {ds: wb[ds]["ci"][0] for ds in DS}
    hi = {ds: wb[ds]["ci"][1] for ds in DS}
    if all(lo[ds] > 0 for ds in DS):
        return "IMPROVES"
    for ds in DS:
        other = [o for o in DS if o != ds][0]
        if lo[ds] > 0 and hi[other] >= 0:
            return "MIXED"
    return "NO"


def load_read(name, ds):
    return np.load(OUT / "reads" / f"{name}__{ds}.npz", allow_pickle=False)


def same_ids(*zs):
    ref = [str(x) for x in zs[0]["ids"]]
    return all([str(x) for x in z["ids"]] == ref for z in zs[1:])


def cell(name, ds, base="mpnet"):
    z, zm, zf = load_read(name, ds), np.load(E0 / base / f"{ds}.npz"), np.load(E0 / "frozen" / f"{ds}.npz")
    if not same_ids(z, zm, zf):
        stop(f"{name} {ds}: the reads hold other questions")
    wb = M.gap(M.per_q(z, "d5"), M.per_q(zm, "d5"))
    e0 = M.gap(M.per_q(zm, "d5"), M.per_q(zf, "d5"))
    return {"r5": round(100 * float(M.per_q(z, "d5").mean()), 2), "won_back": wb,
            "gap": {k: M.gap(M.per_q(z, k), M.per_q(zf, k)) for k in ("d5", "d20", "d1000", "rrf5")},
            "e0_gap": e0["pts"], "share_won": round(wb["pts"] / -e0["pts"], 4) if e0["pts"] < 0 else None}


def cmd_report(a):
    recs = {}
    for arm in ("win", "fuse", "adapt"):
        p = OUT / f"{arm}.json"
        if not p.is_file():
            stop(f"{p.name} missing: every arm reads before the report")
        recs[arm] = json.loads(p.read_text(encoding="utf-8"))
    g1 = {"win": recs["win"]["g1"], "fuse": recs["fuse"]["g1"], "adapt": recs["adapt"]["g1"]}
    flat = [v["pass"] for v in recs["win"]["g1"].values()] + [v["pass"] for e in recs["fuse"]["g1"].values()
                                                               for v in e.values()]
    flat += [v["pass"] for v in recs["adapt"]["g1"].values()] + [v["pass"] for v in recs["win"]["g2"].values()]
    if not all(flat):
        stop(f"G1 or G2 failed: {json.dumps(g1)} {json.dumps(recs['win']['g2'])}")
    rep = {"stage": STAGE, "declared_in": DECLARED, "g1": g1, "g2": recs["win"]["g2"],
           "chosen": {arm: recs[arm]["chosen"] for arm in recs}, "cells": {}, "calls": {}, "beside": {}}
    lines = ["# ENC E2S: can mpnet's vectors be improved without touching mpnet? (report)", "",
             f"Written by `outputs/mp_unified/zenc2s.py report` under {DECLARED}. R@k x100 on s1eval. Won back = the "
             "read minus E0's raw mpnet read, in points; gap = minus E0's frozen read; paired 95% bootstrap over "
             "questions (2,000, seed 0). Settings chosen on s1sel.", "",
             f"Chosen settings: {json.dumps(rep['chosen'])}", "",
             "| arm | carve | R@5 | won back | gap to frozen | share of E0's gap won | R@20 gap | R@1000 gap | RRF@5 gap |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in ("win", "fuse", "adapt"):
        cs = {ds: cell(arm, ds) for ds in DS}
        for ds in DS:
            rep["cells"][f"{arm}__{ds}"] = cs[ds]
            c = cs[ds]
            lines.append(f"| {arm} | {ds} | {c['r5']} | {c['won_back']['pts']} {c['won_back']['ci']} | "
                         f"{c['gap']['d5']['pts']} {c['gap']['d5']['ci']} | {c['share_won']} | {c['gap']['d20']['pts']} | "
                         f"{c['gap']['d1000']['pts']} | {c['gap']['rrf5']['pts']} |")
        call = call_of({ds: cs[ds]["won_back"] for ds in DS})
        bars = [m for m in M.BARS if all(cs[ds]["gap"]["d5"]["pts"] >= M.BARS[m][ds] for ds in DS)]
        rep["calls"][arm] = {"call": call, "bars_pointer": bars}
    hc = cell("adapt", UNSEEN)
    rep["cells"][f"adapt__{UNSEEN}"] = hc
    lines.append(f"| adapt (unseen) | {UNSEEN} | {hc['r5']} | {hc['won_back']['pts']} {hc['won_back']['ci']} | "
                 f"{hc['gap']['d5']['pts']} {hc['gap']['d5']['ci']} | {hc['share_won']} | {hc['gap']['d20']['pts']} | "
                 f"{hc['gap']['d1000']['pts']} | {hc['gap']['rrf5']['pts']} |")
    if hc["won_back"]["ci"][1] < 0:
        rep["calls"]["adapt"]["call"] = "IN-DOMAIN ONLY"
    rep["calls"]["adapt"]["unseen"] = hc["won_back"]["call"]
    for src, dst in (("squad", "musique"), ("musique", "squad")):
        rep["beside"][f"cross-{src}__{dst}"] = cell(f"adapt-cross-{src}", dst)
    for ds in DS:
        z, zf = load_read("adapt-frozen", ds), np.load(E0 / "frozen" / f"{ds}.npz")
        rep["beside"][f"frozen-map__{ds}"] = {"won_back_vs_frozen": M.gap(M.per_q(z, "d5"), M.per_q(zf, "d5"))}
    lines += ["", "Calls (14.5): " + json.dumps({k: v["call"] for k, v in rep["calls"].items()}),
              "", "Beside, a pointer only (13.5's bars met by the gap to frozen): "
              + json.dumps({k: v["bars_pointer"] for k, v in rep["calls"].items()}), "",
              "Beside: cross maps (won back on the other dataset): " + json.dumps(
                  {k: v["won_back"] for k, v in rep["beside"].items() if k.startswith("cross")}), "",
              "Beside: the same map on the frozen vectors (against E0's frozen read): " + json.dumps(
                  {k: v["won_back_vs_frozen"] for k, v in rep["beside"].items() if k.startswith("frozen")}), "",
              "Three arms are read, so the chance that one passes by luck is above 5%."]
    rep["utc"] = utc()
    E.wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {json.dumps({k: v['call'] for k, v in rep['calls'].items()})}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def selftest_toy():
    import torch
    rng = np.random.default_rng(3)
    # topk against a full sort, with ties broken to the lower id
    S = rng.integers(0, 6, size=(7, 40)).astype(np.float32)
    ids, sc = topk(lambda q0, q1: S[q0:q1], 7, 9, qb=3)
    for q in range(7):
        ref = sorted(range(40), key=lambda j: (-S[q, j], j))[:9]
        assert sorted(ids[q].tolist(), key=lambda j: (-S[q, j], j)) == ids[q].tolist()
        assert [S[q, j] for j in ids[q]] == [S[q, j] for j in ref]
    # windows: counts, sizes and the last window's end
    for L in (0, 1, 63, 64, 65, 96, 97, 200, 853):
        offs = [(3 * i, 3 * i + 2) for i in range(L)]
        for W in WINS:
            sp = window_spans(offs, W)
            want = 1 if L <= W else 1 + int(np.ceil((L - W) / (W // 2)))
            assert len(sp) == want, (L, W, len(sp), want)
            if L > W:
                assert sp[-1][1] == offs[-1][1] and all((b - a + 1) == 3 * W for a, b in sp)
                assert all(sp[i][0] < sp[i + 1][0] for i in range(len(sp) - 1))
    # reduceat max equals the per-document maximum
    doc = np.sort(rng.integers(0, 30, 200))
    doc = np.r_[np.arange(30), doc]
    doc.sort()
    Sw = rng.standard_normal((4, doc.size)).astype(np.float32)
    starts = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
    mx = np.maximum.reduceat(Sw, starts, axis=1)
    assert np.array_equal(mx, np.stack([[Sw[q, doc == d].max() for d in range(30)] for q in range(4)]))
    # fuse at the ends is the encoder alone
    Q = {"mpnet": rng.standard_normal((5, 8)).astype(np.float32), "mpnetqa": rng.standard_normal((5, 8)).astype(np.float32)}
    D = {"mpnet": rng.standard_normal((11, 8)).astype(np.float32), "mpnetqa": rng.standard_normal((11, 8)).astype(np.float32)}
    assert np.array_equal(fuse_fn(Q, D, 0.0)(0, 5), Q["mpnet"] @ D["mpnet"].T)
    assert np.allclose(fuse_fn(Q, D, 0.5)(0, 5), 0.5 * (Q["mpnet"] @ D["mpnet"].T + Q["mpnetqa"] @ D["mpnetqa"].T), atol=1e-6)
    # the adapter: epoch 0 is raw mpnet; on a toy problem with a planted rotation the map learns to retrieve
    dim, nd, nq = 32, 400, 3000
    Dm = M.unit(rng.standard_normal((nd, dim)).astype(np.float32))
    R = np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)
    gq = rng.integers(0, nd, nq)
    Qm = M.unit(Dm[gq] @ R + 0.15 * rng.standard_normal((nq, dim)).astype(np.float32))
    for form in FORMS:
        st0 = snapshot(make_g(form, dim))
        assert np.allclose(apply_map(st0, form, Qm[:50]), M.unit(Qm[:50]), atol=1e-6)
    tr, se = np.arange(2500), np.arange(2500, nq)
    golds = [np.asarray([g], np.int64) for g in gq]
    top, _ = topk(lambda q0, q1: Qm[tr][q0:q1] @ Dm.T, tr.size, MINE)
    hard = hard_negs(top, tr, golds)
    assert not any(np.isin(hard[i], golds[r]).any() for i, r in enumerate(tr))
    blk = {"toy": {"Q": Qm[tr], "D": Dm, "pq": np.arange(tr.size, dtype=np.int64), "pd": gq[tr].astype(np.int64),
                   "hard": hard, "keys": np.unique(np.arange(tr.size) * nd + gq[tr])}}
    flat, off = ragged([golds[r] for r in se])
    c = {"flat": flat, "off": off, "gt": np.diff(off)}
    curve, ep, st = train_map(blk, "linear", 1e-2, 3, sel_reader({"toy": (c, Qm[se])}, {"toy": Dm}, "linear"),
                              score=lambda s: s["toy"], batch=256)
    assert ep > 0 and curve[ep]["score"] > curve[0]["score"] + 0.2, curve
    # masking: a batch whose questions share a gold masks that column for the other question
    keys = torch.tensor(sorted([0 * 10 + 4, 1 * 10 + 4, 1 * 10 + 7]))
    qi, di = torch.tensor([0, 1, 1]), torch.tensor([4, 4, 7])
    m = torch.isin(qi[:, None] * 10 + di[None, :], keys)
    m.fill_diagonal_(False)
    assert m.tolist() == [[False, True, False], [True, False, True], [True, True, False]], m.tolist()
    # pick's tie-break and the calls
    cv = [{"w": 0.5, "m": 1.0}, {"w": 0.25, "m": 1.0}, {"w": 0.0, "m": 0.9}]
    assert pick(cv, key=lambda x: x["m"], order=lambda x: (x["w"],))["w"] == 0.25

    def wb(lo, hi):
        return {"ci": [lo, hi]}
    assert call_of({"squad": wb(0.1, 1), "musique": wb(0.2, 2)}) == "IMPROVES"
    assert call_of({"squad": wb(0.1, 1), "musique": wb(-0.2, 0.3)}) == "MIXED"
    assert call_of({"squad": wb(0.1, 1), "musique": wb(-0.9, -0.1)}) == "NO"
    assert call_of({"squad": wb(-0.1, 1), "musique": wb(-0.2, 0.3)}) == "NO"
    return "toy OK"


def selftest_real():
    """Host: the real carves and pairs, windows on real documents, mpnet on the CPU against E1's vectors, and one short
    training run whose epoch 0 equals the raw read."""
    import torch
    torch.set_num_threads(E.THREADS)
    root = E.served_root()
    out = {}
    for ds in DS + (UNSEEN,):
        golds, qids = gold_rows(root, ds)
        cs = sel_carve(ds, golds, qids)
        out[ds] = {"s1sel": len(cs["ids"]), "s1sel_golds": int(cs["flat"].size)}
        if ds == UNSEEN:
            continue
        keep, pq, pd, bad, info = build_pairs(root, ds, golds, cs)
        ce = M.carve(root, ds)
        assert np.isin(ce["flat"], np.asarray(sorted(bad), np.int64)).all(), ds
        assert not np.isin(keep, cs["rows"]).any() and not np.isin(pd, cs["flat"]).any(), ds
        out[ds].update(info)
    m = E.load_model("mpnet", "cpu", E.pinned("mpnet"))
    for ds in DS:
        texts = []
        for _, ch in E.doc_chunks(root, ds, want_shard=lambda s: s == 0):
            if ch is not None:
                texts = ch[:400]
                break
        own = rows_of(root, "mpnet", ds, "docs", np.arange(len(texts)))
        for W in WINS:
            wt, doc = windows_of(m.tokenizer, texts, W)
            lens = [len(x) for x in m.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]]
            want = sum(1 if L <= W else 1 + -(-(L - W) // (W // 2)) for L in lens)
            assert doc.size == want and np.all(np.diff(doc) >= 0), (ds, W, doc.size, want)
            one = np.flatnonzero(np.bincount(doc, minlength=len(texts)) == 1)[:48]
            V, _ = E.encode_texts(m, "mpnet", [wt[int(np.searchsorted(doc, i))] for i in one], "docs")
            cos = np.sum(M.unit(V.astype(np.float32)) * M.unit(own[one]), axis=1)
            out[ds][f"w{W}"] = {"windows_400_docs": int(doc.size), "one_window_min_cos_cpu": round(float(cos.min()), 6)}
            assert cos.min() >= G2_COS, (ds, W, float(cos.min()))
    ds = "squad"
    golds, qids = gold_rows(root, ds)
    cs = sel_carve(ds, golds, qids)
    blk, _, _ = adapt_data(root, "mpnet", ds, golds, cs)
    n = 2000
    sub = {"Q": blk["Q"][:n], "D": blk["D"], "hard": blk["hard"][:n]}
    keep_p = blk["pq"] < n
    sub.update(pq=blk["pq"][keep_p], pd=blk["pd"][keep_p])
    sub["keys"] = np.unique(sub["pq"] * blk["D"].shape[0] + sub["pd"])
    Qs = rows_of(root, "mpnet", ds, "queries", cs["rows"])
    raw = r5_of(topk(lambda q0, q1: Qs[q0:q1] @ blk["D"].T, Qs.shape[0], 5)[0], cs)
    curve, ep, _ = train_map({ds: sub}, "linear", 1e-3, 1, sel_reader({ds: (cs, Qs)}, {ds: blk["D"]}, "linear"),
                             score=lambda s: s[ds])
    assert abs(curve[0]["sel"][ds] - raw) <= 2.0 / len(cs["ids"]), (curve[0], raw)   # unit scaling may flip a near-tie
    out["short_run"] = {"raw_s1sel_r5": round(raw, 6), "curve": curve}
    return out


def cmd_selftest(a):
    print("zenc2s self-test:", selftest_toy(), flush=True)
    if a.real:
        print("zenc2s real:", json.dumps(selftest_real()), flush=True)
    print("zenc2s self-test: OK", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc2s.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fuse", "adapt", "win"):
        p = sub.add_parser(name)
        p.add_argument("--threads", default="8")
    p = sub.add_parser("winenc")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", default="8")
    sub.add_parser("report")
    p = sub.add_parser("selftest")
    p.add_argument("--real", action="store_true")
    p.add_argument("--threads", default="8")
    a = ap.parse_args()
    return {"fuse": cmd_fuse, "adapt": cmd_adapt, "winenc": cmd_winenc, "win": cmd_win, "report": cmd_report,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
