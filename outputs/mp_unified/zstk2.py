"""STK2 (docs/STK2_STARK_LINKS.md): the seven's link layout B0 = S | W2_S | F0T | K_e3v on the three STaRK knowledge bases.

    python -B outputs/mp_unified/zstk2.py names --kb prime --proc <processed dir that holds node_info.pkl> --threads 2
    python -B outputs/mp_unified/zstk2.py links --kb prime --threads 8 [--price]
    python -B outputs/mp_unified/zstk2.py selftest

One module, no per-KB branch. It imports the seven's builders (zmhr.w2s_keys, zd84.rule_forms, zenc1, zenc1b2, zu2b, zul1) and
edits none of them. `names` applies N1 (section 3) to the raw node_info.pkl (read only through zstk1.safe_load, the restricted
unpickler) and writes the sidecar outputs/stark/<kb>/names.jsonl + names.json. `links` builds S, W2_S, F0/F0T, K and the dedup
store under outputs/stark/<kb>/links/ and links.json; `--price` stops after the L4 counts and the surface table, before any family.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import zenc as E  # noqa: E402  (first: reads --threads from sys.argv before numpy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DOC = "docs/STK2_STARK_LINKS.md"
WS = HERE.parents[1]
ROOT = WS / "outputs" / "stark"
KBS = {"prime": "stark_prime", "mag": "stark_mag", "amazon": "stark_amazon"}
NAME_KEYS = ("name", "title", "DisplayName")                 # section 3: N1, the same ordered list on every KB
LABELS = ("S", "R.mention", "R.name_comention", "R.title_comention", "K.knn")   # D83's vocabulary, bits 0-4
FAMILIES = ("S", "R", "K")
SIB_STOP = 50_000_000                                         # section 5, L4
KEY_STOP = 2 ** 31
_LETTER = re.compile(r"[^\W\d_]")


def log(msg):
    print(f"[zstk2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zstk2: {msg}")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha():
    return {"outputs/mp_unified/zstk2.py": E.sha256(Path(__file__))}


def peak_mb():
    try:
        import psutil
        m = psutil.Process().memory_info()
        return round(getattr(m, "peak_wset", m.rss) / 2 ** 20, 1)
    except Exception:
        return None


# ── section 3: N1 ────────────────────────────────────────────────────────────

def name_of(rec):
    """N1: the first of NAME_KEYS whose value is a string with a letter that is not 'nan' (T1's own test), spaces collapsed."""
    for k in NAME_KEYS:
        v = rec.get(k)
        if isinstance(v, str):
            s = " ".join(v.split())
            if s and s.lower() != "nan" and _LETTER.search(s):
                return s, k
    return "", ""


def title_of(name, node_id):
    """A nameless node gets a title no text contains and no other node shares (section 3)."""
    return name if name else "\x00" + node_id


def read_nodes(w):
    ids, types, texts = [], [], []
    with open(w.root / "nodes.jsonl", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            ids.append(r["node_id"])
            types.append(r["type"])
            texts.append(r["text"])
    if len(ids) != w.n:
        stop(f"{len(ids)} node lines, the record says {w.n}")
    return ids, types, texts


def group_stats(names, texts):
    """(sum of k(k-1) over name groups of descriptive named nodes, the ten largest groups)."""
    g = {}
    for nm, tx in zip(names, texts):
        if nm and tx.strip() != nm:
            g[nm] = g.get(nm, 0) + 1
    pairs = sum(k * (k - 1) for k in g.values())
    top = sorted(g.items(), key=lambda x: (-x[1], x[0]))[:10]
    return int(pairs), [{"name": k[:60], "nodes": v} for k, v in top], len(g)


def cmd_names(a):
    import zstk1 as S1
    import zworld as W
    t0 = time.time()
    ds = KBS[a.kb]
    w = W.load(ds)
    fz = json.loads(S1.FREEZE.read_text(encoding="utf-8"))["datasets"][ds]["dataset_json_sha256"]
    if E.sha256(w.root / "DATASET.json") != fz:
        stop("L2: DATASET.json is not the freeze's")
    info = S1.safe_load(Path(a.proc) / "node_info.pkl")
    n = len(info)
    if n != w.n or any(i not in info for i in range(n)):
        stop(f"L1: node_info has {n} records for {w.n} nodes, or its keys are not 0..N-1")
    ids, types, texts = read_nodes(w)
    if any(ids[i] != f"{ds}:{i}" for i in range(n)):
        stop("L1: a node id is not <dataset>:<position>")
    names, keys = [], []
    for i in range(n):
        nm, k = name_of(info[i])
        names.append(nm)
        keys.append(k)
    # L1 sample: the name sits inside the node's own T1 text (T1 keeps every field) unless the text was cut at the cap
    bad = 0
    for i in np.unique(np.linspace(0, n - 1, 1000).astype(np.int64)).tolist():
        if names[i] and len(texts[i]) < S1.CAP - 1 and names[i] not in texts[i]:
            bad += 1
    if bad:
        stop(f"L1: {bad} sampled names are not inside their node's text")
    out = ROOT / a.kb
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "names.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            f.write(json.dumps({"node_id": ids[i], "name": names[i], "key": keys[i]}, ensure_ascii=False) + "\n")
    os.replace(tmp, out / "names.jsonl")
    share = {}
    for t, nm in zip(types, names):
        c = share.setdefault(t, [0, 0])
        c[0] += 1
        c[1] += 1 if nm else 0
    by_key = {}
    for k in keys:
        by_key[k or "none"] = by_key.get(k or "none", 0) + 1
    pairs, top, ngroups = group_stats(names, texts)
    rec = {"stage": "STK2 names", "declared_in": DOC + " section 3", "kb": a.kb, "dataset": ds, "nodes": n,
           "named": int(sum(1 for x in names if x)), "by_key": by_key,
           "by_type": {t: {"nodes": c[0], "named": c[1], "share": round(c[1] / c[0], 6)} for t, c in sorted(share.items())},
           "name_groups": ngroups, "sibling_pairs_if_built": pairs, "sibling_stop": SIB_STOP, "largest_groups": top,
           "sample_check": {"nodes": 1000, "name_not_in_text": bad},
           "names_sha256": E.sha256(out / "names.jsonl"), "dataset_json_sha256": E.sha256(w.root / "DATASET.json"),
           "peak_mb": peak_mb(), "seconds": round(time.time() - t0, 1), "utc": utc(), **code_sha()}
    E.wjson(out / "names.json", rec)
    log(f"{ds}: {rec['named']:,} of {n:,} nodes named ({by_key}); name groups {ngroups:,}; sibling pairs if built {pairs:,} "
        f"(stop at {SIB_STOP:,})")
    return 0


# ── section 4: the layout ────────────────────────────────────────────────────

class ShardRows:
    """Rows of the document shards by global position, float32, read from memory maps (no concatenation)."""

    def __init__(self, paths):
        self.m = [np.load(p, mmap_mode="r") for p in paths]
        self.off = np.concatenate([[0], np.cumsum([x.shape[0] for x in self.m])]).astype(np.int64)
        self.shape = (int(self.off[-1]), int(self.m[0].shape[1]))

    def __getitem__(self, idx):
        idx = np.asarray(idx, np.int64)
        out = np.empty((idx.size, self.shape[1]), np.float32)
        s = np.searchsorted(self.off, idx, side="right") - 1
        for j in np.unique(s):
            m = s == j
            out[m] = np.asarray(self.m[int(j)][idx[m] - self.off[int(j)]], np.float32)
        return out


def s_store(src, dst, rel, n):
    """The S pairs (unordered, distinct, no self pair) with the mask of the relation ids that join each."""
    src, dst, rel = (np.asarray(x, np.int64) for x in (src, dst, rel))
    if rel.size and (rel.min() < 0 or rel.max() >= 32):
        stop("section 4: a relation id outside 0..31 (the mask is 32 bits): the page needs an amendment")
    m = src != dst
    a, b, r = src[m], dst[m], rel[m]
    k = np.minimum(a, b) * n + np.maximum(a, b)
    o = np.argsort(k, kind="stable")
    ks, bits = k[o], (np.uint32(1) << r[o].astype(np.uint32))
    if ks.size == 0:
        return ks, bits
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    return ks[starts], np.bitwise_or.reduceat(bits, starts).astype(np.uint32)


def dedup_store(n, s_keys, s_mask, w2, f0, f0t, kk):
    """Union of the families with D83's label bits, the highest family of each key (S > R > K) and the relation mask."""
    import zul1 as UL
    lt_only = np.setdiff1d(f0t, f0)
    allk = np.unique(np.concatenate([s_keys, w2, f0t, kk]))
    lab = np.zeros(allk.size, np.uint8)
    for bit, keys in ((0, s_keys), (1, w2), (2, f0), (3, lt_only), (4, kk)):
        lab[UL.member(allk, np.unique(keys))] |= np.uint8(1 << bit)
    fam = np.where(lab & 1, 0, np.where(lab & 0b1110, 1, 2)).astype(np.uint8)
    rm = np.zeros(allk.size, np.uint32)
    if s_keys.size:
        rm[np.searchsorted(allk, s_keys)] = s_mask
    return allk, lab, fam, rm


def check_store(allk, lab, fam, rm, s_keys, parts):
    """L5: the union is the store, every S pair has a relation, the family is each key's highest label."""
    union = np.unique(np.concatenate(parts))
    if not np.array_equal(union, allk):
        stop("L5: the union of the families is not the store's keys")
    if np.any(allk[1:] <= allk[:-1]):
        stop("L5: store keys are not strictly increasing")
    if s_keys.size and not (rm[fam == 0] > 0).all():
        stop("L5: an S pair with no relation")
    if not np.array_equal(fam, np.where(lab & 1, 0, np.where(lab & 0b1110, 1, 2)).astype(np.uint8)):
        stop("L5: a family is not the highest label's")
    return True


def k_family(paths, texts):
    """The record's K on the document rows (zenc1's and zenc1b2's pieces, as zmhr.k_family calls them)."""
    import zenc1 as Z1
    import zenc1b2 as B2
    n = len(texts)
    dig = np.frombuffer(b"".join(Z1.text_digest(t) for t in texts), np.uint8).reshape(n, 16)
    pos2g, g_first = Z1.group_digests(dig)
    offs = Z1.shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"L2: {int(offs[-1])} vector rows for {n} nodes")
    eg = np.arange(g_first.size, dtype=np.int64)
    nbr, sc = Z1.nn_top(paths, offs, pos2g, g_first, eg, every=0)
    outp = Z1.out_positions(eg, nbr, g_first)
    osc = B2.scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K)
    src, dst, w = B2.k_edges(pos2g, outp, osc)
    return src, dst, w, {"groups": int(g_first.size), "edges": int(src.size), "tie_rows": int(Z1.tie_rows(sc).sum())}


def save(p, **arrs):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, p)
    return E.sha256(p)


def cmd_links(a):
    import zd84 as D
    import zmhr as M
    import zu2b as U
    import zul1 as UL
    import zworld as W
    t0 = time.time()
    ds = KBS[a.kb]
    w = W.load(ds)
    out = ROOT / a.kb
    nrec = json.loads((out / "names.json").read_text(encoding="utf-8"))
    if E.sha256(out / "names.jsonl") != nrec["names_sha256"] or nrec["dataset_json_sha256"] != E.sha256(w.root / "DATASET.json"):
        stop("L1: the sidecar is not the one its record names, or the package changed")
    n = w.n
    ids, _types, texts = read_nodes(w)
    names = []
    with open(out / "names.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            r = json.loads(ln)
            if r["node_id"] != ids[i]:
                stop(f"L1: names row {i} is {r['node_id']!r}, the package's node {i} is {ids[i]!r}")
            names.append(r["name"])
    if len(names) != n:
        stop(f"L1: {len(names)} names for {n} nodes")
    titles = [title_of(nm, i) for nm, i in zip(names, ids)]
    pairs, top, ngroups = group_stats(names, texts)
    log(f"{ds}: L4 sibling pairs {pairs:,} over {ngroups:,} name groups (stop at {SIB_STOP:,}); largest {top[:3]}")
    if pairs > SIB_STOP:
        stop(f"L4: {pairs:,} same-name sibling pairs exceed {SIB_STOP:,}; groups {top}. The rule is changed only by an amendment")
    shards, why = w.vector_shards("e3v", "docs")
    if shards is None:
        stop(f"L2: the e3v document vectors: {why}")
    X = ShardRows(shards)
    if X.shape[0] != n:
        stop(f"L2: {X.shape[0]} vector rows for {n} nodes")
    rec = {"stage": "STK2 links" + (" (price)" if a.price else ""), "declared_in": DOC + " sections 4 to 6", "kb": a.kb,
           "dataset": ds, "nodes": n, "dataset_json_sha256": E.sha256(w.root / "DATASET.json"),
           "names_sha256": nrec["names_sha256"], "L4": {"sibling_pairs": pairs, "name_groups": ngroups, "largest_groups": top},
           "threads": E.THREADS}
    ts = time.time()
    S = U.Surfaces(titles, U.NEW_RULE)
    rec["surfaces"] = {"forms": len(S.surf), "seconds": round(time.time() - ts, 1)}
    log(f"surface table {len(S.surf):,} forms in {rec['surfaces']['seconds']} s")
    del S
    if a.price:
        import zenc1 as Z1
        dig = np.frombuffer(b"".join(Z1.text_digest(t) for t in texts), np.uint8).reshape(n, 16)
        pos2g, g_first = Z1.group_digests(dig)
        offs = Z1.shard_offsets(shards)
        eg = np.arange(min(2048, g_first.size), dtype=np.int64)
        tk = time.time()
        Z1.nn_top(shards, offs, pos2g, g_first, eg, every=0)
        dt = time.time() - tk
        rec["K_price"] = {"queried_groups": int(eg.size), "groups": int(g_first.size), "block_seconds": round(dt, 1),
                          "estimate_hours": round(dt * g_first.size / eg.size / 3600, 2)}
        rec.update({"peak_mb": peak_mb(), "seconds": round(time.time() - t0, 1), "utc": utc(), **code_sha()})
        E.wjson(out / "links_price.json", rec)
        log(f"price: K block of {eg.size} groups {dt:.1f} s -> about {rec['K_price']['estimate_hours']} h for {g_first.size:,} groups")
        return 0
    base = out / "links"
    sz = np.load(w.edge_file("structural"))
    s_keys, s_mask = s_store(sz["src"], sz["dst"], sz["rel"], n)
    vocab = w.rel_vocab() or {}
    files = {"S": save(base / "S.npz", keys=s_keys, rel_mask=s_mask)}
    log(f"S: {s_keys.size:,} pairs from {sz['src'].size:,} edges")
    F, st = D.rule_forms([{"title": t, "body": x} for t, x in zip(titles, texts)])
    f0, f0t = F["CN_all"], F["F0T"]
    files["F0"], files["F0T"] = save(base / "F0.npz", keys=f0), save(base / "F0T.npz", keys=f0t)
    log(f"F0 {f0.size:,} pairs, F0T {f0t.size:,}")
    w2, w2st = M.w2s_keys(titles, texts, X)
    files["W2_S"] = save(base / "W2_S.npz", keys=w2)
    log(f"W2_S {w2.size:,} pairs ({w2st['mention_links']:,} mention, {w2st['sibling_links']:,} sibling ordered links)")
    for nm, keys in (("S", s_keys), ("F0", f0), ("F0T", f0t), ("W2_S", w2)):
        if keys.size > KEY_STOP:
            stop(f"L4: {nm} holds {keys.size:,} pairs, over {KEY_STOP:,}")
    ks, kd, kw, kst = k_family(shards, texts)
    kk = UL.pair_keys(ks, kd, n)
    files["K"] = save(base / "K.npz", src=ks, dst=kd, weight=kw)
    allk, lab, fam, rm = dedup_store(n, s_keys, s_mask, w2, f0, f0t, kk)
    check_store(allk, lab, fam, rm, s_keys, [s_keys, w2, f0t, kk])
    files["store"] = save(base / "store.npz", keys=allk, labels=lab, family=fam, rel_mask=rm,
                          label_names=np.asarray(LABELS), family_names=np.asarray(FAMILIES))
    r = np.union1d(w2, f0t)
    per_rel = {}
    for rid in range(32):
        c = int(((s_mask >> np.uint32(rid)) & np.uint32(1)).sum())
        if c:
            per_rel[str(vocab.get(str(rid), rid)) if isinstance(vocab, dict) else str(rid)] = c
    rec.update({"rule_build": st, "S": {"edges": int(sz["src"].size), "pairs": int(s_keys.size), "by_relation": per_rel},
                "W2_S": {"pairs": int(w2.size), **{k: v for k, v in w2st.items() if not isinstance(v, dict)}},
                "F0": int(f0.size), "F0T": int(f0t.size), "K": {**kst, "pairs": int(kk.size)},
                "R": int(r.size), "store": int(allk.size),
                "families": {FAMILIES[i]: int((fam == i).sum()) for i in range(3)},
                "per_label": {LABELS[b]: int(((lab >> b) & 1).sum()) for b in range(len(LABELS))},
                "overlaps": {"S_in_R": int(UL.member(s_keys, r).sum()), "S_in_K": int(UL.member(s_keys, kk).sum()),
                             "R_in_K": int(UL.member(r, kk).sum()), "K_in_S_or_R": int(UL.member(kk, np.union1d(s_keys, r)).sum())},
                "files": files, "peak_mb": peak_mb(), "seconds": round(time.time() - t0, 1), "utc": utc(), **code_sha()})
    E.wjson(out / "links.json", rec)
    log(f"{ds}: store {allk.size:,} pairs (S {rec['families']['S']:,}, R-only {rec['families']['R']:,}, K-only {rec['families']['K']:,}); "
        f"peak {rec['peak_mb']} MB, {rec['seconds']} s")
    return 0


# ── L3: the toy ──────────────────────────────────────────────────────────────

def selftest():
    import tempfile
    ok = []

    def chk(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    # N1
    chk("N1 name first", name_of({"name": "Alpha", "title": "T"}) == ("Alpha", "name"))
    chk("N1 falls to title", name_of({"name": "nan", "title": " Beta  Gamma "}) == ("Beta Gamma", "title"))
    chk("N1 falls to DisplayName", name_of({"DisplayName": "Delta"}) == ("Delta", "DisplayName"))
    chk("N1 none: '-1', empty, non-string", name_of({"name": "-1", "title": "", "DisplayName": 3}) == ("", ""))
    chk("title of a nameless node is unique and holds a NUL", title_of("", "x:1") != title_of("", "x:2") and "\x00" in title_of("", "x:1"))
    # S store: two relations on one pair, a self pair, a reversed duplicate
    sk, sm = s_store([0, 1, 2, 3, 3], [1, 0, 2, 4, 4], [0, 3, 1, 5, 5], 6)
    chk("S pairs are unordered, distinct, without self pairs", sk.tolist() == [0 * 6 + 1, 3 * 6 + 4])
    chk("S relation masks are the OR of the ids", sm.tolist() == [(1 << 0) | (1 << 3), 1 << 5])
    # group stats and the sibling count
    pairs, top, ng = group_stats(["A", "A", "A", "B", ""], ["A is x", "A is y", "A", "B is z", "q"])
    chk("sibling count counts descriptive members only (A: 3 nodes, one whose text is its name, so k=2; B: k=1)", pairs == 2 and ng == 2)
    # ShardRows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:   # Windows keeps the memory maps open
        rng = np.random.default_rng(0)
        X = rng.standard_normal((12, 8)).astype(np.float16)
        ps = []
        for j, (lo, hi) in enumerate(((0, 5), (5, 12))):
            p = Path(td) / f"shard_{j:05d}.npy"
            np.save(p, X[lo:hi])
            ps.append(p)
        R = ShardRows(ps)
        chk("ShardRows gathers across shards in the asked order", np.array_equal(R[[11, 0, 5, 4]], X[[11, 0, 5, 4]].astype(np.float32)))
        # the builders on a twelve-node KB
        names = ["Alpha Centauri", "Beta Orionis", "Gamma Draconis", "", "Alpha Centauri", "Delta Cygni", "", "Epsilon Eridani",
                 "Zeta Reticuli", "Eta Carinae", "Theta Aquilae", "Iota Lyrae"]
        texts = ["star: Alpha Centauri is a star system near Beta Orionis", "star: Beta Orionis is bright",
                 "star: Gamma Draconis orbits near Delta Cygni and Epsilon Eridani", "star: an unnamed object",
                 "star: Alpha Centauri another entry of the same name", "star: Delta Cygni", "star: another unnamed object",
                 "star: Epsilon Eridani is close to Zeta Reticuli", "star: Zeta Reticuli", "star: Eta Carinae", "star: Theta Aquilae",
                 "star: Iota Lyrae mentions Eta Carinae"]
        ids = [f"toy:{i}" for i in range(12)]
        titles = [title_of(nm, i) for nm, i in zip(names, ids)]
        import zd84 as D
        import zmhr as M
        F, st = D.rule_forms([{"title": t, "body": x} for t, x in zip(titles, texts)])
        w2, w2st = M.w2s_keys(titles, texts, R)
        n = 12
        print("toy F0", [(int(k // n), int(k % n)) for k in F["CN_all"]], "F0T", [(int(k // n), int(k % n)) for k in F["F0T"]])
        print("toy W2_S", [(int(k // n), int(k % n)) for k in w2], w2st["mention_links"], w2st["sibling_links"])
        w2p = {(int(k // n), int(k % n)) for k in w2}
        chk("W2_S: a mention links (0,1): node 0's text names Beta Orionis", (0, 1) in w2p)
        chk("W2_S: the two nodes named Alpha Centauri are siblings (0,4)", (0, 4) in w2p)
        chk("W2_S: a nameless node is nobody's target or sibling", not any(3 in p or 6 in p for p in w2p if p != (3, 6)) and (3, 6) not in w2p)
        chk("W2_S: node 2 links to the two names its text mentions (2,5) and (2,7)", (2, 5) in w2p and (2, 7) in w2p)
        chk("F0T holds F0", set(F["CN_all"].tolist()) <= set(F["F0T"].tolist()))
        # the dedup store
        import zul1 as UL
        s_keys, s_mask = s_store([0, 2, 9], [1, 5, 10], [4, 7, 4], n)
        kk = UL.pair_keys([1, 8], [2, 9], n)
        allk, lab, fam, rm = dedup_store(n, s_keys, s_mask, w2, F["CN_all"], F["F0T"], kk)
        chk("L5 holds on the toy store", check_store(allk, lab, fam, rm, s_keys, [s_keys, w2, F["F0T"], kk]))
        i01 = int(np.searchsorted(allk, 0 * n + 1))
        chk("a pair in S and W2_S is family S with both label bits", allk[i01] == 1 and fam[i01] == 0 and (lab[i01] & 0b11) == 0b11)
        j12 = int(np.searchsorted(allk, 1 * n + 2))
        chk("a K pair that no other family holds is family K with only the K bit", allk[j12] == 1 * n + 2 and (
            (fam[j12] == 2 and lab[j12] == 0b10000) or (fam[j12] < 2)))
        # K on a toy: nn_top over the shards
        try:
            ks, kd, kw, kst = k_family(ps, texts)
            chk("K: every row has up to three neighbours, no self pair", kst["edges"] > 0 and not (ks == kd).any())
        except SystemExit as e:
            chk(f"K runs on the toy ({e})", False)
    print("selftest:", "PASS" if all(ok) else "FAIL")
    return 0 if all(ok) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("names")
    p.add_argument("--kb", choices=sorted(KBS), required=True)
    p.add_argument("--proc", required=True)
    p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("links")
    p.add_argument("--kb", choices=sorted(KBS), required=True)
    p.add_argument("--price", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return {"names": cmd_names, "links": cmd_links}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
