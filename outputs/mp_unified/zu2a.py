"""U2a (docs/U2A_LINK_GOLD_DIAGNOSTIC.md): which golds the LLM-free links lose, and why. A diagnostic: nothing trains,
nothing is chosen, no filed record changes; outputs only under outputs/u2a/.

    python outputs/mp_unified/zu2a.py selftest
    python outputs/mp_unified/zu2a.py check                  -> outputs/u2a/musique/check.json     (checks 1 and 3)
    python outputs/mp_unified/zu2a.py run --dataset D        -> outputs/u2a/<D>/summary.json, lost_<carve>.jsonl
    python outputs/mp_unified/zu2a.py report                 -> outputs/u2a/report.md

The seeds' GU rows are re-derived on the laptop with u1d.py's own functions (Surfaces' forms, scan, choose_targets),
imported unchanged, over a surface table restricted to the lowercase forms involved (RS). A restricted table finds
exactly the same matches for its surfaces as the full one: u1b.TitleTable matches every surface independently, and
every surface sharing a lowercase form is kept, so the capitalization counts of the kept forms are the full table's.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from array import array
from pathlib import Path
from types import SimpleNamespace

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import u1b  # noqa: E402
import u1d  # noqa: E402

OUT = ROOT / "outputs" / "u2a"
MIN_LEN = u1d.MIN_LEN
CAP = 25                       # the frozen h1_c25 setting's per-seed cap
DAMP_C = 1024                  # U1e's chosen cap (outputs/u1e/choice.json)
CARVES = ("s1sel", "s1eval")
DATASETS = ("hotpotqa", "2wiki")
GRAPHS = ("G0", "GU", "GU_1024")
STEP_ORDER = ("LINKED", "COMMON", "CASE", "ALIAS", "ABSENT", "SHORT")   # most advanced first
BIG = 1 << 40


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ── restricted surfaces ─────────────────────────────────────────────────────


def forms_of(t):
    """u1d.Surfaces' forms of one title, in its order: (surface, kind), kind 0 title, 1 alias, 2 last token."""
    t = t.strip()
    forms = [(t, 0)]
    a = u1d.PAREN.sub("", t).strip()
    if a != t:
        forms.append((a, 1))
    if "," in t:
        b = t.split(",")[0].strip()
        if b != t:
            forms.append((b, 1))
    toks = u1d.PAREN.sub("", t).split()
    if len(toks) >= 2 and toks[-1][:1].isupper():
        forms.append((toks[-1], 2))
    seen, out = set(), []
    for sf, kd in forms:
        if len(sf) < MIN_LEN or (sf, kd) in seen:
            continue
        seen.add((sf, kd))
        out.append((sf, kd))
    return out


class RS:
    """u1d.Surfaces restricted to the forms whose lowercase is in `keep` (every form when keep is None), with global
    node ids: the attributes u1d.scan and u1d.choose_targets read (surf, table, lowid, owners, group)."""

    def __init__(self, titles, keep=None):
        own = {}
        for j, t in enumerate(titles):
            for sf, kd in forms_of(t):
                if keep is None or sf.lower() in keep:
                    own.setdefault(sf, []).append((j, kd))
        self.surf = sorted(own)
        self.index = {s: k for k, s in enumerate(self.surf)}
        ptr, nodes, kinds = [0], [], []
        for sf in self.surf:
            for j, kd in own[sf]:
                nodes.append(j)
                kinds.append(kd)
            ptr.append(len(nodes))
        del own
        self.ptr = np.asarray(ptr, np.int64)
        self.nodes = np.asarray(nodes, np.int64)
        self.kinds = np.asarray(kinds, np.int8)
        lows = {}
        self.lowid = np.asarray([lows.setdefault(s.lower(), len(lows)) for s in self.surf], np.int64)
        self.n_low = len(lows)
        self.low_index = lows
        owners = np.unique(self.nodes)
        self.group = np.full(len(titles), -1, np.int64)
        g = {}
        for j in owners.tolist():
            self.group[j] = g.setdefault(titles[j].strip(), len(g))
        order = owners[np.argsort(self.group[owners], kind="stable")]
        self.g_nodes = order
        self.g_ptr = np.searchsorted(self.group[order], np.arange(len(g) + 1))
        self.n_groups = len(g)
        self.table = u1b.TitleTable(self.surf)
        # each node's own kind-0 surface (its stripped title) in this table, or -1
        self.own_k = np.full(len(titles), -1, np.int64)
        for j in owners.tolist():
            k = self.index.get(titles[j].strip())
            if k is not None:
                self.own_k[j] = k

    def owners(self, k):
        a, b = self.ptr[k], self.ptr[k + 1]
        return self.nodes[a:b], self.kinds[a:b]

    def group_nodes(self, gid):
        return self.g_nodes[self.g_ptr[gid]:self.g_ptr[gid + 1]]


def kind0_lows_in(texts, titles, chunk=400_000):
    """The lowercase forms of every kind-0 surface (stripped title) that u1b.TitleTable matches in any of `texts`."""
    surfs = sorted({t.strip() for t in titles if len(t.strip()) >= MIN_LEN})
    found = set()
    for c in range(0, len(surfs), chunk):
        part = surfs[c:c + chunk]
        tab = u1b.TitleTable(part)
        for text in texts:
            for k in tab.match(-1, text):
                found.add(part[k].lower())
        del tab
    return found


def needed_lows(titles, texts_needed):
    """T: the forms (all kinds) of the needed nodes' titles, and every stripped title their texts name."""
    keep = kind0_lows_in(list(texts_needed.values()), titles)
    for i in texts_needed:
        for sf, _kd in forms_of(titles[i]):
            keep.add(sf.lower())
    return keep


# ── corpus passes ───────────────────────────────────────────────────────────


def iter_texts(path):
    with open(path, "rb") as f:
        for raw in f:
            yield json.loads(raw)["text"]


def read_nodes(served, name, needed):
    """All titles and the texts of `needed`, with nodes.jsonl's sha256 checked against DATASET.json."""
    man = json.loads((served / name / "DATASET.json").read_text(encoding="utf-8"))
    want, n = man["nodes"]["sha256"], int(man["nodes"]["n"])
    h = hashlib.sha256()
    titles, texts = [], {}
    with open(served / name / "nodes.jsonl", "rb") as f:
        for i, raw in enumerate(f):
            h.update(raw)
            rec = json.loads(raw)
            titles.append(u1b.title_of(rec))
            if i in needed:
                texts[i] = rec["text"]
    if len(titles) != n or h.hexdigest() != want:
        raise SystemExit(f"{name}: nodes.jsonl is not DATASET.json's ({len(titles)} lines)")
    return titles, texts, want


def corpus_pass(texts_iter, R, n, watch):
    """One scan of every text with u1d.scan over R: the capitalization counts of R's lowercase forms, the number of texts
    holding a case-ok match of each surface, whether each node's text names its own title (case ok), and, for each
    watched surface, the texts that name it (case ok), ascending."""
    cap = np.zeros(R.n_low, np.int64)
    low = np.zeros(R.n_low, np.int64)
    n_ok = np.zeros(len(R.surf), np.int64)
    self_ok = np.zeros(n, bool)
    lists = {int(k): array("i") for k in watch}
    t0 = time.time()
    x = -1
    for x, text in enumerate(texts_iter):
        hits, st = u1d.scan(R, text)
        for lid, (c, l_) in st.items():
            cap[lid] += c
            low[lid] += l_
        if hits:
            okk = [k for k, ok in hits if ok]
            for k in okk:
                n_ok[k] += 1
                lst = lists.get(k)
                if lst is not None:
                    lst.append(x)
            k0 = R.own_k[x]
            if k0 >= 0 and k0 in okk:
                self_ok[x] = True
        if (x + 1) % 500000 == 0:
            log(f"  corpus pass {x + 1}/{n} ({time.time() - t0:.0f} s)")
    if x + 1 != n:
        raise SystemExit(f"corpus pass read {x + 1} texts, not {n}")
    return cap, low, n_ok, self_ok, lists


class Derived:
    """The V2 links of the needed nodes, re-derived: out (the titles a node's text names) and in (the texts naming its
    title), each through u1d's case gate and capitalization statistic, plus mention in-degrees for U1e's damping."""

    def __init__(self, R, cap, low, n_ok, self_ok, lists, texts):
        self.R, self.n_ok, self.self_ok, self.lists, self.texts = R, n_ok, self_ok, lists, texts
        self.cap, self.low = cap, low
        self.common = low > cap
        self._out = {}

    def out(self, i):
        o = self._out.get(i)
        if o is None:
            R = self.R
            hits, _st = u1d.scan(R, self.texts[i])
            need = []
            res = u1d.choose_targets(R, i, hits, self.common, need)
            tg = set()
            for g in res["V2"]:
                tg.update(R.group_nodes(g).tolist())
            tg.discard(i)
            o = self._out[i] = np.asarray(sorted(tg), np.int64)
        return o

    def inn(self, i):
        k = int(self.R.own_k[i])
        if k < 0 or self.common[self.R.lowid[k]]:
            return np.empty(0, np.int64)
        src = np.frombuffer(self.lists[k], dtype=np.int32).astype(np.int64)
        return src[src != i]

    def indeg(self, j):
        k = int(self.R.own_k[j])
        if k < 0 or self.common[self.R.lowid[k]]:
            return 0
        return int(self.n_ok[k]) - int(self.self_ok[j])

    def rows(self, i):
        """(GU row, GU_1024 row, out, in) of node i: undirected, collapsed, ascending."""
        o, n_ = self.out(i), self.inn(i)
        gu = np.union1d(o, n_)
        od = o[np.asarray([self.indeg(int(j)) <= DAMP_C for j in o], bool)] if o.size else o
        nd = n_ if self.indeg(i) <= DAMP_C else np.empty(0, np.int64)
        return gu, np.union1d(od, nd), o, n_


# ── stores ──────────────────────────────────────────────────────────────────


def row_store(n, rows):
    """A store holding only `rows` ({node: neighbours in the order expand_hops should read them})."""
    nodes = np.asarray(sorted(rows), np.int64)
    counts = np.zeros(n, np.int64)
    if nodes.size:
        counts[nodes] = [rows[int(i)].size for i in nodes]
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    col = np.concatenate([rows[int(i)] for i in nodes]).astype(np.int32) if nodes.size else np.empty(0, np.int32)
    return SimpleNamespace(indptr=indptr, col=col)


def g0_rows(store, nodes):
    """{node: (collapsed row, out-links, in-links)} from M3B's structural store (typed_dir 1: the node is the source)."""
    out = {}
    for s in nodes:
        s = int(s)
        row = store.col[store.indptr[s]:store.indptr[s + 1]].astype(np.int64)
        a, b = store.typed_indptr[s], store.typed_indptr[s + 1]
        tc, td = store.typed_col[a:b].astype(np.int64), store.typed_dir[a:b]
        out[s] = (row, np.unique(tc[td == 1]), np.unique(tc[td == 2]))
    return out


def out_first(o, n_):
    """Out-links first, then the in-links that are not out-links, each ascending."""
    return np.concatenate((o, np.setdiff1d(n_, o))).astype(np.int64)


# ── pools ───────────────────────────────────────────────────────────────────


def tally(pools, golds):
    a = rec = size = 0.0
    for p, g in zip(pools, golds):
        hit = np.isin(g, p)
        a += hit.all()
        rec += hit.mean()
        size += p.size
    n = len(golds)
    return {"ALL": round(a / n, 4), "recall": round(rec / n, 4), "pool": round(size / n, 1)}


def pools_over(mc, mp, store, preps, bases, setting, golds=None):
    """Each carve's pools through the frozen construction's own functions (expand_hops, build_pool); with `golds`, each
    carve's tally instead, without keeping the pools."""
    out = []
    for ci, (pr, bs) in enumerate(zip(preps, bases)):
        pools = []
        a = rec = size = 0.0
        for j, (base, seed) in enumerate(zip(bs, pr.seeds)):
            seed = np.asarray(seed, np.int64)
            exp = mp.expand_hops(seed, [store], setting)
            pool, _ = mc.build_pool(np.asarray(base, np.int64), seed, exp)
            if golds is None:
                pools.append(pool)
            else:
                hit = np.isin(golds[ci][j], pool)
                a += hit.all()
                rec += hit.mean()
                size += pool.size
        if golds is None:
            out.append(pools)
        else:
            nq = len(golds[ci])
            out.append({"ALL": round(a / nq, 4), "recall": round(rec / nq, 4), "pool": round(size / nq, 1)})
    return out


# ── causes ──────────────────────────────────────────────────────────────────


def rule_step(R, common, src_text, tgt_title, D=None, src_i=None, tgt_i=None):
    """How far U1d's V2 rule gets with a link from a source text to a target title (section 4's steps)."""
    t = tgt_title.strip()
    if len(t) < MIN_LEN:
        return "SHORT", {}
    hits, _st = u1d.scan(R, src_text)
    hd = dict(hits)
    k0 = R.index.get(t)
    if k0 is not None and k0 in hd:
        if not hd[k0]:
            return "CASE", {}
        if common[R.lowid[k0]]:
            return "COMMON", {}
        return "LINKED", {}
    found = []
    for sf, kd in forms_of(t):
        if kd == 0:
            continue
        k = R.index.get(sf)
        if k is None or not hd.get(k):
            continue
        nodes, kinds = R.owners(k)
        ent = {"surface": sf, "kind": kd, "common": bool(common[R.lowid[k]])}
        for name, vk in (("k01", (0, 1)), ("k012", (0, 1, 2))):
            m = np.isin(kinds, vk) & (nodes != src_i if src_i is not None else True)
            gs = np.unique(R.group[nodes[m]])
            ent[f"groups_{name}"] = int(gs.size)
            if D is not None and tgt_i is not None and 0 < gs.size <= u1d.AMBIG and R.group[tgt_i] in gs:
                sv = D.get([src_i])[0]
                best, bg = -9.0, -1
                for g in gs.tolist():
                    s = float((D.get(R.group_nodes(g)) @ sv).max())
                    if s > best:
                        best, bg = s, g
                ent[f"context_picks_gold_{name}"] = bool(bg == R.group[tgt_i])
        found.append(ent)
    if found:
        return "ALIAS", {"aliases": found}
    return "ABSENT", {}


def most_advanced(steps):
    return min(steps, key=STEP_ORDER.index) if steps else None


# ── dense ranks ─────────────────────────────────────────────────────────────


def corpus_ranks(D, n, cols, golds, block=768):
    """For column c (a vector) and its gold, 1 + the number of corpus nodes v != gold with c.v/|v| above the gold's."""
    m = len(golds)
    if m == 0:
        return np.empty(0, np.int64)
    B = np.asarray(cols, np.float32)
    golds = np.asarray(golds, np.int64)
    gv = D.get(golds).astype(np.float64)
    sg = np.einsum("ij,ij->i", B.astype(np.float64), gv) / np.linalg.norm(gv, axis=1)
    cnt = np.zeros(m, np.int64)
    t0 = time.time()
    for s in sorted(D.shards):
        p = D.d / D.shards[s]["file"]
        V = np.asarray(np.load(p, mmap_mode="r"), np.float32)
        nv = np.linalg.norm(V, axis=1)
        base = int(s) * u1d.EMB_SHARD
        for c0 in range(0, m, block):
            S = V @ B[c0:c0 + block].T
            S /= nv[:, None]
            gt = S > sg[None, c0:c0 + block]
            cnt[c0:c0 + block] += gt.sum(0)
            own = golds[c0:c0 + block] - base
            idx = np.nonzero((own >= 0) & (own < V.shape[0]))[0]
            if idx.size:
                cnt[c0 + idx] -= gt[own[idx], idx]
            del S, gt
        del V
        if int(s) % 20 == 0:
            log(f"  ranks: shard {s} ({time.time() - t0:.0f} s)")
    return cnt + 1


def row_rank(D, row, vec, gold, chunk=50000):
    """1 + the number of v in row, v != gold, with vec.v/|v| above the gold's."""
    g = D.get([gold])[0].astype(np.float64)
    sg = float(vec.astype(np.float64) @ g / np.linalg.norm(g))
    c = 0
    row = row[row != gold]
    for a in range(0, row.size, chunk):
        V = D.get(row[a:a + chunk])
        s = (V @ vec) / np.linalg.norm(V, axis=1)
        c += int((s > sg).sum())
    return c + 1


# ── run ─────────────────────────────────────────────────────────────────────


def filed_I(name):
    h = json.loads((ROOT / "outputs" / "step4h" / f"coverage_{name}.json").read_text(encoding="utf-8"))
    e = json.loads((ROOT / "outputs" / "u1e" / "c1024" / f"coverage_{name}.json").read_text(encoding="utf-8"))
    out = {}
    for c in CARVES:
        out[c] = {"G0": h["carves"][c]["graphs"]["G0"]["I"], "GU": h["carves"][c]["graphs"]["GU"]["I"],
                  "GU_1024": e["carves"][c]["graphs"]["GU"]["I"], "questions": h["carves"][c]["questions"]}
        if e["carves"][c]["graphs"]["G0"]["I"] != out[c]["G0"]:
            raise SystemExit(f"{name}/{c}: U1e's G0 I differs from step 4h's")
    return out


def cmd_run(a):
    import step4_pool_reach as S4
    import step4c_walk_pools as C4
    t0 = time.time()
    name = a.dataset
    od = OUT / name
    filed = filed_I(name)
    op = S4.Opened()
    mc, mp = op.m3b_compile, S4.m3b_pools
    ds = op.canonical.Dataset(name, root=str(op.served))
    n = int(ds.n_nodes)
    construction = op.construction[name]
    setting = dict(construction["setting"])
    if construction["regime"] != "STRUCT" or int(setting["hops"]) != 1 or int(setting["per_seed_cap"]) != CAP:
        raise SystemExit(f"{name}: not the h1_c25 STRUCT construction")
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c in CARVES]
    if [c for c, _ in pops] != list(CARVES):
        raise SystemExit(f"{name}: carves {[c for c, _ in pops]}")
    if (mc.CSR_CACHE / f"{name}_structural.npz").exists() is False:
        raise SystemExit(f"{name}: M3B's structural store is missing")
    store = mp.load_or_build_store(ds, "structural", mc.CSR_CACHE)
    preps = mc.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, {"structural": store}, op.m3a,
                       op.m3b_contract)
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    bases = [mc.base_rows(construction, pr.dense_ids, pr.splade_ids, op.m3a, op.m3b_contract, constant)
             for pr in preps]
    golds = [[np.unique(np.asarray(g, np.int64)) for g in p.golds] for _c, p in pops]
    log(f"{name}: prepared {[len(p.ids) for _c, p in pops]} questions ({time.time() - t0:.0f} s)")
    seeds_all = np.unique(np.concatenate([np.asarray(s, np.int64) for pr in preps for s in pr.seeds]))
    golds_all = np.unique(np.concatenate([g for gl in golds for g in gl]))
    G0 = g0_rows(store, seeds_all)
    pools = {"G0": [pr.pools for pr in preps]}
    variants = {"G0": {"I_nocap": pools_over(mc, mp, store, preps, bases, {**setting, "per_seed_cap": BIG}, golds),
                       "I_out": pools_over(mc, mp, row_store(n, {s: out_first(r[1], r[2]) for s, r in G0.items()}),
                                           preps, bases, setting, golds)}}
    chk = pools_over(mc, mp, store, preps, bases, setting)
    if any(not np.array_equal(x, y) for c in range(len(pops)) for x, y in zip(chk[c], pools["G0"][c])):
        raise SystemExit(f"{name}: the pools rebuilt over G0 differ from prepare's")
    del store, chk
    log(f"{name}: G0 pools ({time.time() - t0:.0f} s)")

    needed = set(seeds_all.tolist()) | set(golds_all.tolist())
    titles, texts, nodes_sha = read_nodes(op.served, name, needed)
    log(f"{name}: {len(titles)} titles, {len(texts)} needed texts ({time.time() - t0:.0f} s)")
    keep = needed_lows(titles, texts)
    R = RS(titles, keep)
    log(f"{name}: {len(keep)} lowercase forms kept, {len(R.surf)} surfaces, {R.n_groups} title groups "
        f"({time.time() - t0:.0f} s)")
    watch = {int(R.own_k[i]) for i in needed if R.own_k[i] >= 0}
    cap, low, n_ok, self_ok, lists = corpus_pass(iter_texts(op.served / name / "nodes.jsonl"), R, n, watch)
    Dv = Derived(R, cap, low, n_ok, self_ok, lists, texts)
    log(f"{name}: corpus pass done ({time.time() - t0:.0f} s)")
    GU, GD = {}, {}
    for s in seeds_all.tolist():
        gu, gd, o, i_ = Dv.rows(s)
        GU[s], GD[s] = (gu, o, i_), (gd, np.intersect1d(o, gd), np.intersect1d(i_, gd))
    for g, rows in (("GU", GU), ("GU_1024", GD)):
        st_ = row_store(n, {s: r[0] for s, r in rows.items()})
        pools[g] = pools_over(mc, mp, st_, preps, bases, setting)
        variants[g] = {"I_nocap": pools_over(mc, mp, st_, preps, bases, {**setting, "per_seed_cap": BIG}, golds),
                       "I_out": pools_over(mc, mp, row_store(n, {s: out_first(r[1], r[2]) for s, r in rows.items()}),
                                           preps, bases, setting, golds)}
        del st_
    table = {c: {g: {"I": tally(pools[g][ci], golds[ci]), **{v: variants[g][v][ci] for v in variants[g]}}
                 for g in GRAPHS} for ci, c in enumerate(CARVES)}
    checks = {c: {g: {"filed": {k: filed[c][g][k] for k in ("ALL", "recall", "pool")}, "rebuilt": table[c][g]["I"]}
                  for g in GRAPHS} for c in CARVES}
    ok = all(checks[c][g]["filed"] == checks[c][g]["rebuilt"] for c in CARVES for g in GRAPHS) and \
        all(len(golds[ci]) == filed[c]["questions"] for ci, c in enumerate(CARVES))
    summary = {"declared_in": "docs/U2A_LINK_GOLD_DIAGNOSTIC.md", "dataset": name, "nodes_sha256": nodes_sha,
               "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "zu2a_sha256": u1b.sha_src(__file__),
               "u1d_sha256": u1b.sha_src(u1d.__file__), "u1b_sha256": u1b.sha_src(u1b.__file__),
               "kept_lowercase_forms": len(keep), "surfaces": len(R.surf),
               "seeds": int(seeds_all.size), "golds": int(golds_all.size),
               "check2": checks, "status": "CHECKED" if ok else "REFUSED", "pools": table}
    write_json(od / "summary.json", summary)
    log(f"{name}: check 2 {'passes' if ok else 'FAILS'}: " + json.dumps(checks))
    if not ok:
        return 1

    # ── causes ──
    from u1d import Dense
    D = Dense(op.served, name, n)
    fam_rows = {}
    for fam in ("ner", "knn"):
        stf = mp.load_or_build_store(ds, fam, mc.CSR_CACHE)
        fam_rows[fam] = {s: stf.col[stf.indptr[s]:stf.indptr[s + 1]].astype(np.int64) for s in seeds_all.tolist()}
        del stf
    lost_recs = {c: [] for c in CARVES}
    hop_cols, hop_golds, hop_slots = [], [], []
    for ci, c in enumerate(CARVES):
        pop, pr = pops[ci][1], preps[ci]
        for j in range(len(golds[ci])):
            seeds = [int(s) for s in pr.seeds[j]]
            for gd in golds[ci][j].tolist():
                in0 = gd in set(pools["G0"][ci][j].tolist())
                lost = {g: in0 and gd not in set(pools[g][ci][j].tolist()) for g in ("GU", "GU_1024")}
                if not any(lost.values()):
                    continue
                parents = [s for s in seeds if gd in set(G0[s][0][:CAP].tolist())]
                if not parents:
                    raise SystemExit(f"{name}/{c}: lost gold {gd} of question {j} has no G0 parent")
                rec = {"qid": pop.ids[j], "j": j, "gold": gd, "lost": lost, "parents": []}
                for s in parents:
                    dirs = []
                    if gd in set(G0[s][1].tolist()):
                        dirs.append("seed_to_gold")
                    if gd in set(G0[s][2].tolist()):
                        dirs.append("gold_to_seed")
                    steps = []
                    for d_ in dirs:
                        src, tgt = (s, gd) if d_ == "seed_to_gold" else (gd, s)
                        st_, det = rule_step(R, Dv.common, texts[src], titles[tgt], D, src, tgt)
                        steps.append({"dir": d_, "step": st_, **det})
                    pos = {g: (int(np.searchsorted(rows[s][0], gd)) if gd in set(rows[s][0].tolist()) else None,
                               int(rows[s][0].size)) for g, rows in (("GU", GU), ("GU_1024", GD))}
                    rec["parents"].append({"seed": s, "dirs": dirs, "steps": steps,
                                           "row_pos": {g: v[0] for g, v in pos.items()},
                                           "row_size": {g: v[1] for g, v in pos.items()}})
                rec["cause"] = {}
                for g, rows in (("GU", GU), ("GU_1024", GD)):
                    if not lost[g]:
                        continue
                    if any(gd in set(rows[s][0].tolist()) for s in seeds):
                        rec["cause"][g] = "CAP_CUT"
                    elif g == "GU_1024" and any(gd in set(GU[s][0].tolist()) for s in seeds):
                        rec["cause"][g] = "DAMPED"
                    else:
                        rec["cause"][g] = most_advanced([x["step"] for p_ in rec["parents"] for x in p_["steps"]])
                for fam in ("ner", "knn"):
                    rec[f"{fam}_any"] = any(gd in set(fam_rows[fam][s].tolist()) for s in seeds)
                    rec[f"{fam}_25"] = any(gd in set(fam_rows[fam][s][:CAP].tolist()) for s in seeds)
                for m_, ids in (("dense", pr.dense_ids[j]), ("splade", pr.splade_ids[j])):
                    w = np.nonzero(np.asarray(ids) == gd)[0]
                    rec[f"{m_}_rank"] = int(w[0]) + 1 if w.size else None
                q = np.asarray(pr.qemb[j], np.float32)
                rec["hop"] = []
                for s in parents:
                    p_ = D.get([s])[0].astype(np.float64)
                    p_ /= np.linalg.norm(p_)
                    r = (q.astype(np.float64) - (q.astype(np.float64) @ p_) * p_).astype(np.float32)
                    h = {"parent": s}
                    row = GU[s][0]
                    if gd in set(row.tolist()):
                        h["rank_in_gu_row_residual"] = row_rank(D, row, r, gd)
                        h["rank_in_gu_row_question"] = row_rank(D, row, q, gd)
                        h["gu_row_size"] = int(row.size)
                    hop_cols.append(r)
                    hop_golds.append(gd)
                    hop_slots.append((c, len(lost_recs[c]), len(rec["hop"]), "rank_corpus_residual"))
                    rec["hop"].append(h)
                hop_cols.append(q)
                hop_golds.append(gd)
                hop_slots.append((c, len(lost_recs[c]), None, "rank_corpus_question"))
                lost_recs[c].append(rec)
        log(f"{name}/{c}: {len(lost_recs[c])} lost golds classified ({time.time() - t0:.0f} s)")
    ranks = corpus_ranks(D, n, hop_cols, hop_golds)
    for (c, ri, hi, key), rk in zip(hop_slots, ranks.tolist()):
        if hi is None:
            lost_recs[c][ri][key] = rk
        else:
            lost_recs[c][ri]["hop"][hi][key] = rk
    log(f"{name}: corpus ranks done ({time.time() - t0:.0f} s)")
    causes = {}
    for ci, c in enumerate(CARVES):
        with open(od / f"lost_{c}.jsonl", "w", encoding="utf-8") as f:
            for rec in lost_recs[c]:
                f.write(json.dumps(rec) + "\n")
        n_golds = int(sum(g.size for g in golds[ci]))
        causes[c] = {"golds": n_golds}
        for g in ("GU", "GU_1024"):
            recs = [r for r in lost_recs[c] if r["lost"][g]]
            cnt = {}
            for r in recs:
                cnt[r["cause"][g]] = cnt.get(r["cause"][g], 0) + 1
            nl = len(recs)
            ent = {"lost": nl, "lost_share_of_golds": round(nl / n_golds, 4),
                   "causes": {k: {"n": v, "share_of_lost": round(v / max(nl, 1), 4),
                                  "share_of_golds": round(v / n_golds, 4)} for k, v in sorted(cnt.items())}}
            for fam in ("ner", "knn"):
                ent[f"{fam}_any"] = round(sum(r[f"{fam}_any"] for r in recs) / max(nl, 1), 4)
                ent[f"{fam}_25"] = round(sum(r[f"{fam}_25"] for r in recs) / max(nl, 1), 4)
            best = [min(h["rank_corpus_residual"] for h in r["hop"]) for r in recs]
            qr = [r["rank_corpus_question"] for r in recs]
            for k in (25, 100, 1000):
                ent[f"residual_corpus_top{k}"] = round(sum(b <= k for b in best) / max(nl, 1), 4)
                ent[f"question_corpus_top{k}"] = round(sum(b <= k for b in qr) / max(nl, 1), 4)
            cc = [r for r in recs if r["cause"][g] == "CAP_CUT"]
            if cc:
                rr = [min(h.get("rank_in_gu_row_residual", BIG) for h in r["hop"]) for r in cc]
                rq = [min(h.get("rank_in_gu_row_question", BIG) for h in r["hop"]) for r in cc]
                ent["cap_cut_within_25_by_residual"] = round(sum(x <= CAP for x in rr) / len(cc), 4)
                ent["cap_cut_within_25_by_question"] = round(sum(x <= CAP for x in rq) / len(cc), 4)
            causes[c][g] = ent
    summary.update({"causes": causes, "seconds": round(time.time() - t0, 1), "status": "RUN"})
    write_json(od / "summary.json", summary)
    log(f"{name}: done ({time.time() - t0:.0f} s): " + json.dumps(causes)[:2000])
    return 0


# ── check (musique) ─────────────────────────────────────────────────────────


def cmd_check(_a):
    t0 = time.time()
    served, freeze = u1b.package_root(False)
    name = "musique"
    link = u1d.OUT / name / "link"
    rec = json.loads((link / "s0of1.json").read_text(encoding="utf-8"))
    if sha_file(link / "s0of1.npz") != rec["npz_sha256"]:
        raise SystemExit("musique link shard: sha256 differs from its record")
    z = np.load(link / "s0of1.npz")
    src, dst = z["V2_src"].astype(np.int64), z["V2_dst"].astype(np.int64)
    n = int(rec["n_nodes"])
    ind = np.bincount(dst, minlength=n)
    top = np.argsort(-ind, kind="stable")[:50]
    needed = set(range(0, n, 37)) | set(top.tolist())
    titles, texts, nodes_sha = read_nodes(served, name, needed)
    if nodes_sha != rec["nodes_sha256"]:
        raise SystemExit("musique: nodes differ from U1d's link run")
    # check 3: the unrestricted table is u1d.Surfaces
    S = u1d.Surfaces(titles)
    R0 = RS(titles, None)
    own = np.unique(R0.nodes)
    fw, bw, part_ok = {}, {}, True
    for a_, b_ in zip(S.group[own].tolist(), R0.group[own].tolist()):
        if fw.setdefault(a_, b_) != b_ or bw.setdefault(b_, a_) != a_:
            part_ok = False
            break
    c3 = {"surf": S.surf == R0.surf, "ptr": bool(np.array_equal(S.ptr, R0.ptr)),
          "nodes": bool(np.array_equal(S.nodes, R0.nodes)), "kinds": bool(np.array_equal(S.kinds, R0.kinds)),
          "lowid": bool(np.array_equal(S.lowid, R0.lowid)), "groups": part_ok and len(fw) == R0.n_groups}
    del R0
    # check 1: the restricted re-derivation equals the filed V2 edges and capstat
    keep = needed_lows(titles, texts)
    R = RS(titles, keep)
    watch = {int(R.own_k[i]) for i in needed if R.own_k[i] >= 0}
    cap, low, n_ok, self_ok, lists = corpus_pass(iter_texts(served / name / "nodes.jsonl"), R, n, watch)
    Dv = Derived(R, cap, low, n_ok, self_ok, lists, texts)
    order = np.argsort(src, kind="stable")
    s_sorted, d_by_src = src[order], dst[order]
    order2 = np.argsort(dst, kind="stable")
    d_sorted, s_by_dst = dst[order2], src[order2]
    bad_out = bad_in = bad_deg = 0
    for i in sorted(needed):
        a_, b_ = np.searchsorted(s_sorted, [i, i + 1])
        fo = np.unique(d_by_src[a_:b_])
        a_, b_ = np.searchsorted(d_sorted, [i, i + 1])
        fi = np.unique(s_by_dst[a_:b_])
        bad_out += not np.array_equal(fo, Dv.out(i))
        bad_in += not np.array_equal(fi, Dv.inn(i))
        bad_deg += int(ind[i]) != Dv.indeg(i)
    cs = json.loads((u1d.OUT / name / "capstat.json").read_text(encoding="utf-8"))
    if sha_file(u1d.OUT / name / "capstat.npz") != cs["npz_sha256"]:
        raise SystemExit("musique capstat: sha256 differs from its record")
    zc = np.load(u1d.OUT / name / "capstat.npz")
    full_low = {S.surf[k].lower(): int(S.lowid[k]) for k in range(len(S.surf))}
    bad_cap = 0
    for lw, lid in R.low_index.items():
        fl = full_low[lw]
        bad_cap += int(zc["cap"][fl]) != int(cap[lid]) or int(zc["low"][fl]) != int(low[lid])
    c1 = {"nodes_examined": len(needed), "out_differ": int(bad_out), "in_differ": int(bad_in),
          "indegree_differ": int(bad_deg), "forms_compared": len(R.low_index), "capstat_differ": int(bad_cap),
          "kept_lowercase_forms": len(keep), "surfaces": len(R.surf)}
    ok = all(c3.values()) and not (bad_out or bad_in or bad_deg or bad_cap)
    res = {"declared_in": "docs/U2A_LINK_GOLD_DIAGNOSTIC.md", "dataset": name, "check1": c1, "check3": c3,
           "status": "PASS" if ok else "FAIL", "freeze_RECORD_SHA256": freeze, "zu2a_sha256": u1b.sha_src(__file__),
           "u1d_sha256": u1b.sha_src(u1d.__file__), "link_record_u1d_sha256": rec["u1d_sha256"],
           "seconds": round(time.time() - t0, 1)}
    write_json(OUT / name / "check.json", res)
    log(json.dumps(res))
    return 0 if ok else 1


# ── report ──────────────────────────────────────────────────────────────────


def cmd_report(_a):
    lines = ["# U2a: which golds the LLM-free links lose, and why", "",
             "Generated by `outputs/mp_unified/zu2a.py report` from `outputs/u2a/`. Declared in "
             "docs/U2A_LINK_GOLD_DIAGNOSTIC.md.", ""]
    ck = OUT / "musique" / "check.json"
    if ck.exists():
        c = json.loads(ck.read_text(encoding="utf-8"))
        lines += [f"Checks 1 and 3 (musique): **{c['status']}**: {json.dumps(c['check1'])}; {json.dumps(c['check3'])}",
                  ""]
    for name in DATASETS:
        p = OUT / name / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        lines += [f"## {name} ({s['status']})", "", "| carve | graph | I | I_nocap | I_out |", "| --- | --- | --- | --- | --- |"]
        for c in CARVES:
            for g in GRAPHS:
                t = s["pools"][c][g]
                lines.append(f"| {c} | {g} | " + " | ".join(f"{t[v]['ALL']:.4f} / {t[v]['recall']:.4f} / {t[v]['pool']:.1f}"
                                                         for v in ("I", "I_nocap", "I_out")) + " |")
        lines.append("")
        for c in CARVES:
            cz = s.get("causes", {}).get(c)
            if not cz:
                continue
            for g in ("GU", "GU_1024"):
                e = cz[g]
                lines.append(f"- {c} {g}: {e['lost']} lost golds ({e['lost_share_of_golds']:.4f} of {cz['golds']}): " +
                             ", ".join(f"{k} {v['n']} ({v['share_of_lost']:.2f})" for k, v in e["causes"].items()) +
                             f"; ner any {e['ner_any']:.2f}, knn any {e['knn_any']:.2f}; residual hop top-25 "
                             f"{e['residual_corpus_top25']:.2f}, top-100 {e['residual_corpus_top100']:.2f}; question "
                             f"top-100 {e['question_corpus_top100']:.2f}" +
                             (f"; CAP_CUT within 25 by question {e['cap_cut_within_25_by_question']:.2f}, by residual "
                              f"{e['cap_cut_within_25_by_residual']:.2f}" if "cap_cut_within_25_by_question" in e else ""))
        lines.append("")
    p = OUT / "report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    print("\n".join(lines))
    return 0


# ── selftest ────────────────────────────────────────────────────────────────


def cmd_selftest(_a):
    titles = ["Paris", "Paris (mythology)", "London Bridge", "John Smith", "Smith, Adam", "Apple", "Apple", "The Band",
              "Rome", "abc"]
    texts = ["Paris is a city. Paris and London Bridge appear; the band played.",
             "In myth, Paris took Helen. John Smith wrote of Paris.",
             "London Bridge spans the river. paris is lower here. The Band toured.",
             "John Smith met Smith in Rome. An apple fell.",
             "Adam Smith, Smith wrote. Apple pie. Rome.",
             "Apple is a company. apple apple apple.",
             "Apple records. Rome and Paris.",
             "The Band. the band of Rome.",
             "Rome is old. Paris (mythology) is a figure.",
             "abc"]
    n = len(titles)
    S = u1d.Surfaces(titles)
    R0 = RS(titles, None)
    assert S.surf == R0.surf and np.array_equal(S.ptr, R0.ptr) and np.array_equal(S.nodes, R0.nodes)
    assert np.array_equal(S.kinds, R0.kinds) and np.array_equal(S.lowid, R0.lowid)
    # full-table reference: u1d's stat and link logic over every text
    capf = np.zeros(S.n_low, np.int64)
    lowf = np.zeros(S.n_low, np.int64)
    for t in texts:
        _h, st = u1d.scan(S, t)
        for lid, (c, l_) in st.items():
            capf[lid] += c
            lowf[lid] += l_
    common_f = lowf > capf
    ref = set()
    for i, t in enumerate(texts):
        hits, _ = u1d.scan(S, t)
        need = []
        res = u1d.choose_targets(S, i, hits, common_f, need)
        tg = set()
        for g in res["V2"]:
            tg.update(S.group_nodes(g).tolist())
        tg.discard(i)
        ref.update((i, j) for j in tg)
    needed = {0, 3, 6}
    tn = {i: texts[i] for i in needed}
    keep = needed_lows(titles, tn)
    R = RS(titles, keep)
    watch = {int(R.own_k[i]) for i in needed if R.own_k[i] >= 0}
    cap, low, n_ok, self_ok, lists = corpus_pass(iter(texts), R, n, watch)
    full_low = {S.surf[k].lower(): int(S.lowid[k]) for k in range(len(S.surf))}
    for lw, lid in R.low_index.items():
        assert capf[full_low[lw]] == cap[lid] and lowf[full_low[lw]] == low[lid], lw
    Dv = Derived(R, cap, low, n_ok, self_ok, lists, tn)
    ind = {}
    for (i, j) in ref:
        ind[j] = ind.get(j, 0) + 1
    for i in needed:
        assert set(Dv.out(i).tolist()) == {j for (a_, j) in ref if a_ == i}, (i, Dv.out(i))
        assert set(Dv.inn(i).tolist()) == {a_ for (a_, j) in ref if j == i}, (i, Dv.inn(i))
        assert Dv.indeg(i) == ind.get(i, 0), (i, Dv.indeg(i), ind.get(i, 0))
    # damping against u1e's rule
    import u1e
    es = np.asarray(sorted(ref), np.int64)
    if es.size:
        keep_e, _m = u1e.damp(es[:, 0], es[:, 1], np.zeros(len(es), np.int16), 0, n, 1)
        kept = {tuple(x) for x in es[keep_e].tolist()}
        global DAMP_C
        old, DAMP_C = DAMP_C, 1
        try:
            for i in needed:
                gu, gd, o, i_ = Dv.rows(i)
                want = {j for (a_, j) in kept if a_ == i} | {a_ for (a_, j) in kept if j == i}
                assert set(gd.tolist()) == want, (i, gd, want)
                assert set(gu.tolist()) == {j for (a_, j) in ref if a_ == i} | {a_ for (a_, j) in ref if j == i}
        finally:
            DAMP_C = old
    # rule steps
    common = Dv.common
    assert rule_step(R, common, "nothing here", "abc")[0] == "SHORT"
    assert rule_step(R, common, "we saw ROME there", "Rome")[0] in ("CASE", "ABSENT")
    assert rule_step(R, common, "the river at London Bridge.", "London Bridge")[0] == "LINKED"
    assert rule_step(R, common, "No mention at all.", "London Bridge")[0] == "ABSENT"
    # out_first, row_store, tally
    assert out_first(np.array([5, 9]), np.array([1, 5, 7])).tolist() == [5, 9, 1, 7]
    st = row_store(12, {3: np.array([4, 1, 9]), 7: np.array([2])})
    assert st.col[st.indptr[3]:st.indptr[4]].tolist() == [4, 1, 9] and st.col[st.indptr[7]:st.indptr[8]].tolist() == [2]
    assert tally([np.array([1, 2, 3])], [np.array([2, 5])]) == {"ALL": 0.0, "recall": 0.5, "pool": 3.0}
    assert most_advanced(["ABSENT", "CASE", "SHORT"]) == "CASE"
    # ranks against brute force
    rng = np.random.default_rng(0)
    V = rng.normal(size=(300, 1536)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    tmpd = OUT / "_selftest"
    tmpd.mkdir(parents=True, exist_ok=True)
    np.save(tmpd / "shard_00000.npy", V.astype(np.float16))
    Df = SimpleNamespace(d=tmpd, shards={0: {"file": "shard_00000.npy"}},
                         get=lambda rows: np.asarray(np.load(tmpd / "shard_00000.npy"), np.float32)[np.asarray(rows)])
    cols = rng.normal(size=(4, 1536)).astype(np.float32)
    gl = [3, 50, 299, 7]
    got = corpus_ranks(Df, 300, cols, gl)
    V16 = np.asarray(np.load(tmpd / "shard_00000.npy"), np.float32)
    for c, g, r in zip(cols, gl, got.tolist()):
        s = (V16 @ c) / np.linalg.norm(V16, axis=1)
        want = 1 + int(np.sum(np.delete(s, g) > s[g]))
        assert abs(r - want) <= 1, (r, want)
        assert abs(row_rank(Df, np.arange(300), c, g) - want) <= 1
    for p in tmpd.iterdir():
        p.unlink()
    tmpd.rmdir()
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=DATASETS)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    return {"selftest": cmd_selftest, "check": cmd_check, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
