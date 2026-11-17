"""D91 (docs/D91_D83_WITHOUT_LABELS.md): D83's graph with the record's three families in place of D83's five rule labels.

D83 links each pool pair once per rule that made it (zd83.LABELS: S, R.mention, R.name_comention, R.title_comention,
K.knn). D91 links the same pairs once per look family, as zlink does on the record (0 the slot, 1 the NER store, 2 the knn
store), on D83's stores (F0T, the S > R > K dedup). D83's look chunks are pruned, so D91's family links are derived from
D83's labelled links (derive_part), and nothing upstream of the links is rebuilt: D83's files are hard-linked (assemble).

Why the derivation is exact. After D83's dedup the families' stores are:
  KB (metaqa, webqsp): slot = S; ner = F0T minus S; knn = knn minus (S + ner). Disjoint, so a linked pair's family is
    the one store that holds it: S -> 0, else R.* -> 1, else K.knn -> 2.
  Passages: slot = W2_S (R.mention); ner = F0T (R.name + R.title), no dedup against the slot; knn = knn minus (slot + F0T).
    A pair with only one of the slot and NER labels is in that family only; a K.knn pair with neither is in knn only.
    A pair in both the slot and F0T has a family-f edge when the look kept it: the look keeps, per pool node, its first
    CAP (64) in-pool neighbours of each family (m3b_features.typed_pool_edges and weighted_pool_edges; both stores hold
    both directions). If a node's in-pool family degree were above CAP, the look would give it CAP family edges, so it
    would carry at least CAP links with that family's label. So an end with at most CAP - 1 labelled links of the family
    has all its in-pool family neighbours as edges, and the pair has the family edge. Only a pair whose two ends both
    carry CAP or more links of a family's label is undecided for that family ("open"); `resolve` decides each from the
    look's own edge function on D83's stores (G2d), and G1d checks every passage part against the look's message count.

    python outputs/mp_unified/zd91.py arm --arm D91F --dataset hotpotqa --host              (one carve: links, typed, record)
    python outputs/mp_unified/zd91.py arm --arm D91F --dataset hotpotqa --host --dry --parts 1   (checks only; no file)
    python outputs/mp_unified/zd91.py gate --arm D91F --host
    python outputs/mp_unified/zd91.py survey --arm D83E --dataset hotpotqa                  (counts only: open pairs)
    python outputs/mp_unified/zd91.py selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

DECLARED = "docs/D91_D83_WITHOUT_LABELS.md"
STAGE = "D91"
NMAX = 1 << 15                       # zlink.NMAX: pool-local rows are int16
CAP = 64                             # m3b_features.IN_POOL_CAP: a pool node's family edges at most
LABELS = ("S", "R.mention", "R.name_comention", "R.title_comention", "K.knn")   # zd83.LABELS, ids 0 to 4
FAMS = 3                             # zlink.FAMS: the record's families
KB = ("metaqa", "webqsp")
SRC_ARMS = {"D83F": "outputs/abl/D83F", "D83E": "outputs/abl/D83E", "b1v": "outputs/b1v"}
ARMS = {"D91F": "D83F", "D91E": "D83E", "D91B": "b1v"}                  # D91's arm: the D83 arm it is derived from
# every D91 arm under outputs/abl/D91*: outputs/b1x, the first name of D91B, is AP2's carve folder (17 Nov 05:25, moved out; claim())
OUT_OF = {a: ROOT / "outputs" / "abl" / a for a in ARMS}
CARVES = {"D91F": {"squad": "fit", "hotpotqa": "fit", "2wiki": "fit", "musique": "s1fit", "webqsp": "fit"},   # zd83.ARMS
          "D91E": {d: "s1eval" for d in ("musique", "metaqa", "hotpotqa", "squad", "2wiki", "webqsp")},
          "D91B": {d: "b1v" for d in ("2wiki", "hotpotqa", "musique")}}       # D91B keeps b1v's carve name and files
LINKED = ("pools", "look", "cache", "chains", "kt", "d83kt", "settings")   # D83's folders D91 takes as they are (section 2)
BENCH_V = ROOT / "outputs" / "b1v" / "bench" / "hipporag2_v"            # b1v's setting files (zd83b.BENCH_V)


def log(msg):
    print(f"[zd91 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def write_json(p, obj):
    """Through a temporary name and os.replace, so a reader never sees half a record."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save_npy(d, k, x):
    """np.save's bytes under d/k.npy (through a temporary name); returns the file's sha256 (records name file shas)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{k}.{os.getpid()}.tmp.npy"
    np.save(tmp, np.ascontiguousarray(x))
    os.replace(tmp, d / f"{k}.npy")
    return sha_file(d / f"{k}.npy")


def under(p, base):
    """G3d: D91 writes only under its own arm."""
    p, base = Path(p).resolve(), Path(base).resolve()
    if base != p and base not in p.parents:
        raise SystemExit(f"D91 G3d: {p} is outside {base}")
    return p


# ── the derivation ───────────────────────────────────────────────────────────

def pair_runs(q_ne, e_u, e_v, e_lab):
    """D83's links (sorted by question, u, v, label; unique) grouped by pair: (question, u, v, label bits) per pair."""
    q_ne = np.asarray(q_ne, np.int64)
    u, v, lab = (np.asarray(x, np.int64) for x in (e_u, e_v, e_lab))
    if u.size != int(q_ne.sum()) or v.size != u.size or lab.size != u.size:
        raise SystemExit(f"D91: {u.size} links against q_ne {int(q_ne.sum())}")
    if lab.size and (int(lab.min()) < 0 or int(lab.max()) >= len(LABELS)):
        raise SystemExit("D91: a link label outside 0..4")
    q = np.repeat(np.arange(q_ne.size, dtype=np.int64), q_ne)
    key = (q * NMAX + u) * NMAX + v
    full = key * len(LABELS) + lab
    if full.size > 1 and not (np.diff(full) > 0).all():
        raise SystemExit("D91: D83's links are not sorted and unique by (question, u, v, label)")
    start = np.r_[True, key[1:] != key[:-1]] if key.size else np.zeros(0, bool)
    idx = np.nonzero(start)[0]
    bits = np.bitwise_or.reduceat(np.left_shift(1, lab), idx) if idx.size else np.zeros(0, np.int64)
    return q[idx], u[idx], v[idx], bits.astype(np.int64), q_ne.size


def _deg(q, u, m, nq):
    """Per pair end (q, u): how many pairs with mask m that end has (the pairs hold both directions)."""
    k = q[m] * NMAX + u[m]
    uk, cnt = np.unique(k, return_counts=True)
    return uk, cnt


def _deg_at(uk, cnt, k):
    if uk.size == 0:
        return np.zeros(k.size, np.int64)
    i = np.searchsorted(uk, k)
    i[i >= uk.size] = uk.size - 1
    return np.where(uk[i] == k, cnt[i], 0)


def derive_part(q_ne, e_u, e_v, e_lab, kb, cap=CAP):
    """One block's pairs from D83's labelled links: (question, u, v, label bits, family bits decided, family bits open
    (1 << f), questions, counts). `emit` turns decided bits into zlink's links; `resolve` decides the open ones first."""
    q, u, v, bits, nq = pair_runs(q_ne, e_u, e_v, e_lab)
    slot = 0 if kb else 1
    sb = (bits >> slot) & 1 == 1
    nb = (bits & 0b01100) != 0                                   # R.name_comention or R.title_comention
    kn = (bits >> 4) & 1 == 1
    if kb:
        other_slot = (bits >> 1) & 1 == 1                        # R.mention never on a KB
        if other_slot.any():
            raise SystemExit("D91: an R.mention link on a KB")
        f0, f1, f2 = sb, nb & ~sb, kn & ~sb & ~nb
        open0 = open1 = np.zeros(q.size, bool)
    else:
        if ((bits & 1) != 0).any():
            raise SystemExit("D91: an S link on a passage dataset")
        both = sb & nb
        d0 = _deg(q, u, sb, nq)
        d1 = _deg(q, u, nb, nq)
        ku, kv = q * NMAX + u, q * NMAX + v
        open0 = both & (_deg_at(*d0, ku) >= cap) & (_deg_at(*d0, kv) >= cap)
        open1 = both & (_deg_at(*d1, ku) >= cap) & (_deg_at(*d1, kv) >= cap)
        f0 = sb & ~open0
        f1 = nb & ~open1
        f2 = kn & ~sb & ~nb
    fb = f0.astype(np.int64) | (f1.astype(np.int64) << 1) | (f2.astype(np.int64) << 2)
    ob = open0.astype(np.int64) | (open1.astype(np.int64) << 1)
    none = (fb == 0) & (ob == 0)
    if none.any():
        raise SystemExit(f"D91: {int(none.sum())} pairs with no family (a label outside its family's store)")
    st = {"pairs": int(q.size), "pairs_slot_and_ner": int((sb & nb).sum()) if not kb else 0,
          "open_slot": int(open0.sum()), "open_ner": int(open1.sum()), "open_pairs": int((ob != 0).sum())}
    return q, u, v, bits, fb, ob, nq, st


def emit(q, u, v, fb, nq):
    """Family links (zlink's form) from the pairs' family bits: one link per (question, u, v, family), sorted."""
    fam = np.concatenate([np.full(int(((fb >> f) & 1).sum()), f, np.int64) for f in range(FAMS)])
    src = np.concatenate([np.nonzero((fb >> f) & 1)[0] for f in range(FAMS)])
    order = np.lexsort((fam, src))
    pi, fam = src[order], fam[order]
    qq, a, b = q[pi], u[pi], v[pi]
    ne = np.bincount(qq, minlength=nq).astype(np.int64)
    key = ((qq * NMAX + a) * NMAX + b) * FAMS + fam
    if key.size > 1 and not (np.diff(key) > 0).all():
        raise SystemExit("D91: the family links are not sorted and unique")
    return ne, a.astype(np.int16), b.astype(np.int16), fam.astype(np.int8)


def messages(ne, a, f, cap=CAP):
    """G1d: the look's message-edge count the family links imply, the sum over questions, pool nodes and families of
    min(CAP, the node's family links) (the links hold both directions; a passage store holds no self-loop)."""
    qq = np.repeat(np.arange(np.asarray(ne).size, dtype=np.int64), ne)
    k = (qq * NMAX + np.asarray(a, np.int64)) * FAMS + np.asarray(f, np.int64)
    _u, c = np.unique(k, return_counts=True)
    return int(np.minimum(c, cap).sum())


# ── open pairs: the look's own edge function on D83's stores (G2d) ───────────

def family_edges(store, pool, fam, cap=CAP):
    """A pool's family-fam message edges as the look makes them (m3b_features.pool_edges: the lookup table over the
    store's node universe; typed_pool_edges for the slot, weighted_pool_edges for NER), as sorted unique undirected
    keys min * NMAX + max of pool-local rows, self-loops dropped."""
    from mp_retrieval import m3b_features as MF
    pool = np.ascontiguousarray(pool, np.int64)
    lookup = MF._acquire_lookup(pool, int(store.n_nodes))
    try:
        if fam == 0:
            _e, _pid, (a, b) = MF.typed_pool_edges(store, pool, cap, lookup)
        else:
            a, b, _w = MF.weighted_pool_edges(store, pool, cap, lookup)
    finally:
        MF._release_lookup(lookup, pool)
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    k = a != b
    return np.unique(np.minimum(a[k], b[k]) * NMAX + np.maximum(a[k], b[k]))


def resolve(q, u, v, bits, fb, ob, pools, stores, cap=CAP):
    """The open pairs of a passage block decided by the look's edge function on each question's pool (pools(i): its
    rows as global ids, in the look's order) and D83's store of the family (stores.get(f)). G2d per question: every
    recomputed edge is a D83 link with the family's label, and every pair already decided for the family is one of
    them. Returns the family bits and counts; stops if a pair is left with no family."""
    fb = np.array(fb, np.int64, copy=True)
    st = {"open_questions": 0, "slot_kept": 0, "slot_not": 0, "ner_kept": 0, "ner_not": 0}
    if not (np.asarray(ob) != 0).any():
        return fb, st
    pk = np.minimum(u, v) * NMAX + np.maximum(u, v)
    lab = {0: ((bits >> 1) & 1) == 1, 1: (bits & 0b01100) != 0}
    qs = np.unique(q[ob != 0])
    starts, ends = np.searchsorted(q, qs, "left"), np.searchsorted(q, qs, "right")
    for i, s, e in zip(qs.tolist(), starts.tolist(), ends.tolist()):
        pool = pools(i)
        st["open_questions"] += 1
        for f, nm in ((0, "slot"), (1, "ner")):
            of = ((ob[s:e] >> f) & 1) == 1
            if not of.any():
                continue
            E = family_edges(stores.get(f), pool, f, cap)
            lf = lab[f][s:e]
            if E.size and int(E.max()) >= int(pool.size) * NMAX:
                raise SystemExit(f"D91 G2d: question {i}: an edge outside its pool of {pool.size}")
            have = np.unique(pk[s:e][lf])
            if not np.isin(E, have, assume_unique=True).all():
                raise SystemExit(f"D91 G2d: question {i}: a recomputed {nm} edge is not a D83 link with the {nm} label")
            dec = lf & ~of
            if not np.isin(pk[s:e][dec], E).all():
                raise SystemExit(f"D91 G2d: question {i}: a pair decided for the {nm} family is not a recomputed edge")
            kept = of & np.isin(pk[s:e], E)
            fb[s:e] |= kept.astype(np.int64) << f
            st[f"{nm}_kept"] += int(kept.sum())
            st[f"{nm}_not"] += int((of & ~kept).sum())
    if (fb == 0).any():
        raise SystemExit(f"D91 G2d: {int((fb == 0).sum())} pairs in no family after the open pairs are decided")
    return fb, st


def label_sets(src_arm, ds):
    """D83's five label key sets of a dataset (written by D83's look step), each checked against its record."""
    d = ROOT / SRC_ARMS[src_arm] / "labels" / ds
    rec = read_json(d / "record.json")
    if rec.get("labels") != list(LABELS) or rec.get("dataset") != ds:
        raise SystemExit(f"D91: {rel(d)} is not D83's label record of {ds}")
    keys = []
    for i, lab in enumerate(LABELS):
        f = d / f"{i}.npy"
        if sha_file(f) != rec["sha256"][lab]:
            raise SystemExit(f"D91: {rel(f)} is not the file its record names")
        keys.append(np.load(f, allow_pickle=False))
    return keys, int(rec["nodes"]), sha_file(d / "record.json")


class Stores:
    """D83's slot and NER stores of one passage dataset, built on first use (only a carve with open pairs needs them)
    and gated (G2d): the slot's pairs are D83's R.mention set, NER's are R.name_comention plus R.title_comention."""

    def __init__(self, src_arm, ds, host):
        self.src_arm, self.ds, self.host, self.made, self.info = src_arm, ds, host, {}, None

    def get(self, f):
        if not self.made:
            self.build()
        return self.made[f]

    def build(self):
        import d83stores as DS
        import zb1p as B
        t = time.time()
        keys, n, lab_sha = label_sets(self.src_arm, self.ds)
        M = B.mods(self.host)
        if self.src_arm == "b1v":
            LB = M.LB
            LB.BENCH = BENCH_V                                      # zd83b.bind's folder: b1v's setting files
            st = LB.load_setting(self.ds)
            slot = LB.setting_store(M.S4.m3b_pools, st, "structural", n)
            ner = LB.setting_store(M.S4.m3b_pools, st, "ner", n)
            how = {"slot": "look_b1.setting_store(structural) on b1v's setting", "ner": "look_b1.setting_store(ner)",
                   "setting_build_sha256": sha_file(BENCH_V / self.ds / "build.json")}
        else:
            import zd83
            ip, col, _desc = M.U2C.graph_csr(self.ds, "chosen", n)    # zabl.on_arm's slot R on a passage dataset: W2_S
            slot = M.U2C.csr_store(n, np.asarray(ip, np.int64), np.asarray(col, np.int32))
            del ip, col
            a, b, w, _f0k = zd83.f0t_pairs(self.ds, n)
            ner = DS.weighted_store("ner", n, a, b, w)
            del a, b, w
            how = {"slot": "zu2c.csr_store of zu2c.graph_csr(chosen)", "ner": "d83stores.weighted_store of zd83.f0t_pairs"}
        g_slot = bool(np.array_equal(DS.pair_keys(slot), keys[1]))
        g_ner = bool(np.array_equal(DS.pair_keys(ner), np.union1d(keys[2], keys[3])))
        if not (g_slot and g_ner):
            raise SystemExit(f"D91 G2d: {self.src_arm}/{self.ds}: the rebuilt stores are not D83's (slot {g_slot}, ner {g_ner})")
        self.made = {0: slot, 1: ner}
        self.info = dict(how, nodes=n, label_sets_record_sha256=lab_sha, slot_pairs=int(keys[1].size),
                         ner_pairs=int(np.union1d(keys[2], keys[3]).size), g2d_stores="EQUAL",
                         seconds=round(time.time() - t, 1))
        log(f"stores {self.src_arm}/{self.ds}: {self.info}")


# ── the arm: hard links, family links, typed sets, carve record ──────────────

def part_dirs(base, ds, carve):
    """A folder's parts (part_IofN with a record.json) in index order."""
    d = Path(base) / ds / carve
    ps = [p for p in d.iterdir() if p.is_dir() and (p / "record.json").exists()] if d.exists() else []

    def key(p):
        m = re.search(r"(\d+)of(\d+)", p.name)
        return (int(m.group(2)), int(m.group(1))) if m else (0, 0)
    return sorted(ps, key=key)


def mine(parts, ds):
    """A file of D83's LINKED folders that belongs to dataset ds (or the cache's bases, which every carve shares)."""
    name = parts[-1]
    if ds in parts[1:-1] or name.startswith(f"{ds}__") or name.split(".")[0] == ds:
        return True
    return parts[0] == "cache" and len(parts) == 2 and name.startswith("basis_")


def link_one(src, dst):
    """dst as a hard link of src; a file already at dst must be the same file or hold the same bytes (G3d)."""
    if dst.exists():
        if os.path.samefile(src, dst):
            return "same"
        if sha_file(src) == sha_file(dst):
            return "equal"
        raise SystemExit(f"D91 G3d: {rel(dst)} is there and is not {rel(src)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except FileExistsError:
        return link_one(src, dst)
    return "linked"


def assemble(arm, ds):
    """Section 2's equal part: D83's files of the dataset in LINKED, hard-linked into D91's arm (never rewritten)."""
    S, O = ROOT / SRC_ARMS[ARMS[arm]], OUT_OF[arm]
    cnt, nbytes = {}, 0
    for top in LINKED:
        base = S / top
        if not base.exists():
            continue
        for dp, _dn, fn in os.walk(base):
            for f in fn:
                src = Path(dp) / f
                parts = src.relative_to(S).parts
                if ".tmp" in f or not mine(parts, ds):
                    continue
                r = link_one(src, under(O.joinpath(*parts), O))
                cnt[r] = cnt.get(r, 0) + 1
                nbytes += src.stat().st_size
    log(f"assemble {arm}/{ds}: {cnt} ({nbytes / 1e9:.2f} GB, hard links)")
    return {"files": cnt, "bytes": nbytes}


def load_checked(d, rec, keys, mmap=False):
    out = {}
    for k in keys:
        f = Path(d) / f"{k}.npy"
        if sha_file(f) != rec["arrays"][k]["sha256"]:
            raise SystemExit(f"D91 G3d: {rel(f)} is not the file its record names")
        x = np.load(f, mmap_mode="r" if mmap else None, allow_pickle=False)
        if x.dtype.str != rec["arrays"][k]["dtype"] or list(x.shape) != rec["arrays"][k]["shape"]:
            raise SystemExit(f"D91: {rel(f)} is {x.dtype} {x.shape}, its record says otherwise")
        out[k] = x
    return out


def zl_settings(h83):
    """zlink's settings (the record's: load_links refuses any other), checked against D83's minus the labels."""
    import zlink as ZL
    s = dict(ZL.SETTINGS)
    d83 = {k: v for k, v in h83["settings"].items() if k not in ("families", "labels", "edges")}
    if d83 != {k: v for k, v in s.items() if k not in ("families", "edges")} or s["families"] != FAMS:
        raise SystemExit(f"D91: D83's link settings {h83['settings']} are not zlink's {s} but for the labels")
    return s


def build_links(arm, ds, host=False, dry=False, max_parts=None):
    """Section 4 step 2 (and 3 on a KB): per cache part, D91's family links from D83's labelled links, open pairs
    decided, G1d on passage parts; on a KB the typed sets moved to the family-0 links. Returns the per-part summary."""
    src_arm = ARMS[arm]
    S, O, carve, kb = ROOT / SRC_ARMS[src_arm], OUT_OF[arm], CARVES[arm][ds], ds in KB
    cparts, lparts = part_dirs(S / "cache", ds, carve), part_dirs(S / "links", ds, carve)
    if not cparts or [p.name for p in cparts] != [p.name for p in lparts]:
        raise SystemExit(f"D91: {src_arm}/{ds}/{carve}: cache parts {[p.name for p in cparts]}, links {[p.name for p in lparts]}")
    stores = None if kb else Stores(src_arm, ds, host)
    settings = None
    tdir83 = S / "typed" / ds / carve
    trec83 = read_json(tdir83 / "record.json") if kb else None
    if kb and [*trec83["parts"]] != [p.name for p in cparts]:
        raise SystemExit(f"D91: {rel(tdir83)}: parts {list(trec83['parts'])}")
    summary, typed_parts, tot = {}, {}, {}
    for pi, (cp, lp) in enumerate(zip(cparts, lparts)):
        if max_parts is not None and pi >= max_parts:
            break
        t = time.time()
        crec_sha = sha_file(cp / "record.json")
        crec = read_json(cp / "record.json")
        h83_sha = sha_file(lp / "record.json")
        h83 = read_json(lp / "record.json")
        if h83["step1_record_sha256"] != crec_sha or h83["queries"] != crec["queries"] or h83["rows"] != crec["rows"]:
            raise SystemExit(f"D91: {rel(lp)} is not D83's links of {rel(cp)}")
        if h83["settings"].get("labels") != list(LABELS) or "d83" not in h83:
            raise SystemExit(f"D91: {rel(lp)} is not a D83 labelled links record")
        settings = settings or zl_settings(h83)
        od = under(O / "links" / ds / carve / lp.name, O)
        if not dry and (od / "record.json").exists():
            h91 = read_json(od / "record.json")
            if h91.get("d91", {}).get("d83_links_record_sha256") == h83_sha and h91["step1_record_sha256"] == crec_sha:
                load_checked(od, h91, ("q_ne", "e_u", "e_v", "e_fam"), mmap=True)
                if not kb or (under(O / "typed" / ds / carve / lp.name, O) / "record.json").exists():
                    summary[lp.name] = h91["d91"]["counts"]
                    add_counts(tot, {k_: v_ for k_, v_ in summary[lp.name].items() if k_ not in ("families", "g1d")})
                    if kb:
                        typed_parts[lp.name] = sha_file(O / "typed" / ds / carve / lp.name / "record.json")
                    log(f"links {arm}/{ds}/{carve}/{lp.name}: filed, its arrays checked")
                    continue
        arr = load_checked(lp, h83, ("q_ne", "e_u", "e_v", "e_fam"), mmap=True)
        node = np.load(cp / "node.npy", mmap_mode="r")
        nn = np.load(cp / "n.npy").astype(np.int64)
        for k_ in ("node", "n"):
            if "arrays" in crec and k_ in crec["arrays"] and sha_file(cp / f"{k_}.npy") != crec["arrays"][k_]["sha256"]:
                raise SystemExit(f"D91 G3d: {rel(cp / (k_ + '.npy'))} is not the file its record names")
        if nn.size != h83["queries"] or int(nn.sum()) != h83["rows"] or node.size != h83["rows"]:
            raise SystemExit(f"D91: {rel(cp)}: n and node do not match D83's links record")
        off = np.r_[0, np.cumsum(nn)]
        acc = {"q_ne": [], "e_u": [], "e_v": [], "e_fam": []}
        st, msgs = {}, 0
        for q0, q1, l0, l1 in blocks(arr["q_ne"]):
            q, u, v, bits, fb, ob, nq, sb = derive_part(arr["q_ne"][q0:q1], arr["e_u"][l0:l1], arr["e_v"][l0:l1],
                                                         arr["e_fam"][l0:l1], kb)
            if not kb:
                fb, rs = resolve(q, u, v, bits, fb, ob,
                                 lambda i, q0=q0: np.asarray(node[off[q0 + i]:off[q0 + i + 1]], np.int64), stores)
                sb.update(rs)
            ne, a, b, f = emit(q, u, v, fb, nq)
            if not kb:
                msgs += messages(ne, a, f)
            add_counts(st, sb)
            for k_, x in zip(acc, (ne, a, b, f)):
                acc[k_].append(x)
            del q, u, v, bits, fb, ob
        arrays = {k_: np.concatenate(x) for k_, x in acc.items()}
        del acc
        E91 = int(arrays["e_u"].size)
        le = h83["look_edges"]
        if kb:
            g1d = {"verdict": "NOT_APPLICABLE", "why": "KB: the stores are disjoint (exact); S holds self-loops"}
            k83, m83 = _keys_of(arr["q_ne"], arr["e_u"], arr["e_v"], np.asarray(arr["e_fam"]) == 0)
            k91, m91 = _keys_of(arrays["q_ne"], arrays["e_u"], arrays["e_v"], arrays["e_fam"] == 0)
            if not np.array_equal(k83, k91):
                raise SystemExit(f"D91: {rel(lp)}: the family-0 links are not D83's S links")
            del k83, k91
        else:
            want = int(le["raw"]) - int(le["self_loops"])
            g1d = {"messages": msgs, "look_raw": int(le["raw"]), "self_loops": int(le["self_loops"]),
                   "verdict": "EQUAL" if (msgs == want and int(le["self_loops"]) == 0) else "DIFF"}
            if g1d["verdict"] != "EQUAL":
                raise SystemExit(f"D91 G1d: {rel(lp)}: {g1d}")
        fams = np.bincount(arrays["e_fam"].astype(np.int64), minlength=FAMS)
        counts = dict(st, links=E91, d83_links=int(arr["e_u"].size), families=[int(x) for x in fams], g1d=g1d)
        add_counts(tot, {k_: v_ for k_, v_ in counts.items() if k_ not in ("families", "g1d")})
        summary[lp.name] = counts
        log(f"links {arm}/{ds}/{carve}/{lp.name}: {E91:,} links of {int(arr['e_u'].size):,} D83 links; families "
            f"{counts['families']}; open {st.get('open_pairs', 0)} (slot kept {st.get('slot_kept', 0)}, not "
            f"{st.get('slot_not', 0)}; ner kept {st.get('ner_kept', 0)}, not {st.get('ner_not', 0)}); G1d {g1d['verdict']}"
            f"{'' if kb else ' ' + str(msgs)} ({time.time() - t:.0f} s)")
        if dry:
            continue
        shas = {k_: save_npy(od, k_, x) for k_, x in arrays.items()}
        rec = {"dataset": ds, "carve": carve, "part": lp.name, "step1_record_sha256": crec_sha, "chunks": h83["chunks"],
               "queries": int(arrays["q_ne"].size), "rows": h83["rows"], "edges": E91,
               "edges_per_row": round(E91 / max(1, h83["rows"]), 3), "families": counts["families"], "look_edges": le,
               "look": h83["look"], "carve_ids_sha256": h83["carve_ids_sha256"], "n_against_step1": h83["n_against_step1"],
               "settings": settings,
               "arrays": {k_: {"dtype": v_.dtype.str, "shape": list(v_.shape), "sha256": shas[k_]} for k_, v_ in arrays.items()},
               "placement": {"where": "host" if host else "laptop"}, "script_sha256": sha_src(__file__),
               "seconds": round(time.time() - t, 1), "utc": utc(),
               "d91": {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "from_arm": src_arm,
                       "d83_links_record_sha256": h83_sha, "d83_labels": h83["d83"]["labels"],
                       "label_sets_record_sha256": h83["d83"].get("label_sets_record_sha256"), "counts": counts,
                       "stores": stores.info if stores is not None and stores.made else None}}
        write_json(od / "record.json", rec)
        if kb:
            typed_parts[lp.name] = move_typed(arm, ds, carve, cp, lp, od, crec_sha, h83_sha, arr, arrays, tdir83, trec83)
    if kb and not dry and (max_parts is None or max_parts >= len(cparts)):
        typed_carve(arm, ds, carve, tdir83, trec83, typed_parts, summary)
    return summary, tot


def _keys_of(q_ne, e_u, e_v, m):
    q_ne = np.asarray(q_ne, np.int64)
    q = np.repeat(np.arange(q_ne.size, dtype=np.int64), q_ne)
    i = np.nonzero(np.asarray(m))[0]
    return (q[i] * NMAX + np.asarray(e_u)[i].astype(np.int64)) * NMAX + np.asarray(e_v)[i].astype(np.int64), i


def move_typed(arm, ds, carve, cp, lp, od, crec_sha, h83_sha, arr83, arr91, tdir83, trec83):
    """Section 4 step 3: one KB part's typed sets and direction bits moved from D83's S links to D91's family-0 links
    (the same (question, u, v) in the same order); every other link carries none, as on D83's other labels."""
    O = OUT_OF[arm]
    tp83 = tdir83 / lp.name
    if sha_file(tp83 / "record.json") != trec83["parts"][lp.name]:
        raise SystemExit(f"D91: {rel(tp83)}: not the record {rel(tdir83)} names")
    h = read_json(tp83 / "record.json")
    if h["step1_record_sha256"] != crec_sha or h["link_record_sha256"] != h83_sha:
        raise SystemExit(f"D91: {rel(tp83)}: not built on {rel(cp)} and {rel(lp)}")
    t83 = load_checked(tp83, h, ("t_set", "t_dir"))
    m83 = np.asarray(arr83["e_fam"]) == 0
    m91 = arr91["e_fam"] == 0
    if t83["t_set"].size != m83.size or int(m83.sum()) != int(m91.sum()):
        raise SystemExit(f"D91: {rel(tp83)}: {t83['t_set'].size} typed rows against {m83.size} D83 links")
    if (t83["t_set"][~m83] != -1).any() or (t83["t_dir"][~m83] != 0).any():
        raise SystemExit(f"D91: {rel(tp83)}: a link that is not S carries a typed set")
    t_set = np.full(m91.size, -1, np.int16)
    t_dir = np.zeros(m91.size, np.int8)
    t_set[m91], t_dir[m91] = t83["t_set"][m83], t83["t_dir"][m83]
    td = under(O / "typed" / ds / carve / lp.name, O)
    shas = {"t_set": save_npy(td, "t_set", t_set), "t_dir": save_npy(td, "t_dir", t_dir)}
    rec = dict(h, link_record_sha256=sha_file(od / "record.json"), edges=int(m91.size),
               arrays={k: {"dtype": x.dtype.str, "shape": list(x.shape), "sha256": shas[k]}
                       for k, x in (("t_set", t_set), ("t_dir", t_dir))},
               placement={"where": "host"}, utc=utc(),
               d91={"declared_in": DECLARED, "stage": STAGE, "arm": arm, "from_record_sha256": trec83["parts"][lp.name],
                    "moved": int(m91.sum()), "typed": int((t_set >= 0).sum()), "script_sha256": sha_src(__file__)})
    write_json(td / "record.json", rec)
    return sha_file(td / "record.json")


def typed_carve(arm, ds, carve, tdir83, trec83, typed_parts, summary):
    O = OUT_OF[arm]
    td = under(O / "typed" / ds / carve, O)
    link_one(tdir83 / "sets.npy", td / "sets.npy")                    # the same bytes: the sets are D83's
    if sha_file(td / "sets.npy") != trec83["sets"]["sha256"]:
        raise SystemExit(f"D91: {rel(td / 'sets.npy')} is not the sets file D83's record names")
    rec = dict(trec83, parts=typed_parts, edges=int(sum(c["links"] for c in summary.values())), utc=utc(),
               d91={"declared_in": DECLARED, "stage": STAGE, "arm": arm,
                    "from_record_sha256": sha_file(tdir83 / "record.json"), "script_sha256": sha_src(__file__)})
    write_json(td / "record.json", rec)
    log(f"typed {arm}/{ds}/{carve}: {len(typed_parts)} part(s), {rec['edges']:,} edges")


def record_step(arm, ds, host=False):
    """Section 4 step 4: PKO's carve record of D91's carve (zabl's and zb1h's record step: record_carve with the source
    arm's step-1 note). A record already filed is re-checked, never rewritten."""
    import zb1p as B
    M = B.mods(host)
    O, carve = OUT_OF[arm], CARVES[arm][ds]
    R = M.PK.roots(O)
    out = R.records / f"{ds}__{carve}.json"
    if out.exists():
        rec = read_json(out)
        bad = [f for f, s in rec["inputs"].items() if not (O / f).exists() or sha_file(O / f) != s]
        if bad:
            raise SystemExit(f"D91: {rel(out)} is filed and {len(bad)} of its inputs changed ({bad[:3]})")
        log(f"record {arm}/{ds}: filed, its {len(rec['inputs'])} inputs checked")
        return rec
    src = read_json(ROOT / SRC_ARMS[ARMS[arm]] / "records" / f"{ds}__{carve}.json")
    note = json.loads(json.dumps(src["step1_check"]))
    return M.PK.record_carve(ds, carve, R, step1=lambda *_a: dict(note))


CLAIM = "d91_arm.json"
CLAIM_TOPS = set(LINKED) | {"links", "typed", "records", "steps", "logs", "ledger.json", "gate.json", "d91_gate.json", CLAIM}


def claim(arm, O=None):
    """The arm's folder is D91's before anything is written there: absent or empty (claimed now), claimed by this arm already,
    or holding only D91's folders and records of the arm's own carves (claimed now). Anything else stops the row: a folder of
    another stage is never written into (G3d)."""
    O = Path(O) if O is not None else OUT_OF[arm]
    if O.parent.name != "abl" or not O.name.startswith("D91"):
        raise SystemExit(f"D91 G3d: {rel(O)} is not a D91 arm folder (outputs/abl/D91*)")
    mark = O / CLAIM
    if mark.exists():
        got = read_json(mark)
        if got.get("arm") != arm or got.get("stage") != STAGE:
            raise SystemExit(f"D91 G3d: {rel(O)} is claimed by {got.get('stage')}/{got.get('arm')}, not D91/{arm}")
        return got
    if O.exists():
        tops = sorted(p.name for p in O.iterdir())
        odd = [x for x in tops if x not in CLAIM_TOPS]
        mine_recs = {f"{d}__{c}.json" for d, c in CARVES[arm].items()}
        recs = sorted(p.name for p in (O / "records").glob("*.json")) if (O / "records").exists() else []
        if odd or any(r not in mine_recs for r in recs):
            raise SystemExit(f"D91 G3d: {rel(O)} holds what D91 does not write ({odd[:5]}, records {recs[:5]}): another stage's folder")
    got = {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "from_arm": ARMS[arm], "carves": CARVES[arm], "utc": utc()}
    write_json(under(mark, O), got)
    return got


def arm_row(arm, ds, host=False, dry=False, max_parts=None):
    if ds not in CARVES[arm]:
        raise SystemExit(f"D91: {arm} has no carve on {ds} ({sorted(CARVES[arm])})")
    t = time.time()
    import zb1p as B
    B.mods(host)                                             # the host mirror first (zpkoh), as every D83 step
    if not dry:
        claim(arm)
    linked = None if dry else assemble(arm, ds)
    summary, tot = build_links(arm, ds, host, dry, max_parts)
    if dry:
        log(f"dry {arm}/{ds}: {tot}")
        return 0
    rec = record_step(arm, ds, host)
    O, carve = OUT_OF[arm], CARVES[arm][ds]
    write_json(under(O / "steps" / f"{ds}__{carve}__d91.json", O),
               {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "from_arm": ARMS[arm], "dataset": ds, "carve": carve,
                "linked": linked, "parts": summary, "total": tot, "record_inputs": len(rec["inputs"]),
                "record_sha256": sha_file(O / "records" / f"{ds}__{carve}.json"), "zd91_sha256": sha_src(__file__),
                "seconds": round(time.time() - t, 1), "utc": utc()})
    log(f"arm {arm}/{ds}/{carve}: {tot} ({time.time() - t:.0f} s)")
    return 0


def gate(arm, host=False):
    """PKO's gate over the arm's carves (every input each record names re-hashed; section 5's checks as D83's gate)."""
    claim(arm)
    import zb1p as B
    M = B.mods(host)
    O = OUT_OF[arm]
    R = M.PK.roots(O)
    carves = list(CARVES[arm].items())
    missing = [f"{d}__{c}" for d, c in carves if not (R.records / f"{d}__{c}.json").exists()]
    if missing:
        raise SystemExit(f"D91 {arm}: no record yet for {missing}")
    with B.checks(M):
        verdict = M.PK.gate(R, carves=carves)
    write_json(O / "d91_gate.json", {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "pko_gate": rel(R.gate),
                                      "pko_gate_sha256": sha_file(R.gate), "verdict": verdict, "carves": carves,
                                      "zd91_sha256": sha_src(__file__), "utc": utc()})
    return 0


# ── survey: counts only ──────────────────────────────────────────────────────

def links_parts(arm, ds):
    base = ROOT / SRC_ARMS[arm] / "links" / ds
    if not base.exists():
        raise SystemExit(f"D91: no links under {base}")
    carves = [p for p in base.iterdir() if p.is_dir()]
    if len(carves) != 1:
        raise SystemExit(f"D91: {base} holds {len(carves)} carves")
    return carves[0].name, sorted(p for p in carves[0].iterdir() if (p / "record.json").exists())


BLOCK_LINKS = 20_000_000              # D83 links per block of whole questions (bounded memory; pairs never cross questions)


def blocks(q_ne, limit=BLOCK_LINKS):
    """[(q0, q1, link0, link1)] blocks of whole questions holding at most `limit` links (one question may hold more)."""
    q_ne = np.asarray(q_ne, np.int64)
    off = np.r_[0, np.cumsum(q_ne)]
    out, q0 = [], 0
    while q0 < q_ne.size:
        q1 = int(np.searchsorted(off, off[q0] + limit, side="right")) - 1
        q1 = max(q1, q0 + 1)
        q1 = min(q1, q_ne.size)
        out.append((q0, q1, int(off[q0]), int(off[q1])))
        q0 = q1
    return out


def add_counts(tot, st):
    for k_, x in st.items():
        if isinstance(x, int) and not isinstance(x, bool):
            tot[k_] = tot.get(k_, 0) + x


def survey(arm, ds):
    carve, parts = links_parts(arm, ds)
    tot = {}
    out = {"arm": arm, "dataset": ds, "carve": carve, "parts": {}}
    for p in parts:
        t = time.time()
        arr = {k: np.load(p / f"{k}.npy", mmap_mode="r") for k in ("q_ne", "e_u", "e_v", "e_fam")}
        st = {}
        for q0, q1, l0, l1 in blocks(arr["q_ne"]):
            *_x, sb = derive_part(arr["q_ne"][q0:q1], arr["e_u"][l0:l1], arr["e_v"][l0:l1], arr["e_fam"][l0:l1], ds in KB)
            add_counts(st, sb)
        st["seconds"] = round(time.time() - t, 1)
        out["parts"][p.name] = st
        add_counts(tot, st)
        log(f"survey {arm}/{ds}/{carve}/{p.name}: {st}")
    out["total"] = tot
    print(json.dumps(out["total"]))
    return out


# ── self-test ────────────────────────────────────────────────────────────────

def selftest():
    """The derivation and the open-pair decisions against the look itself: random stores (a typed structural slot,
    weighted NER and knn), unsorted pools, m3b_features.pool_edges as the look, d83stores.label_edges for both D83's
    labelled links and zlink's family links. D91's links must be zlink's bit for bit at every cap, and G1d must hold."""
    from types import SimpleNamespace
    import d83stores as DS
    from mp_retrieval import m3b_features as MF
    from mp_retrieval.m3b_pools import FamilyStore
    rng = np.random.default_rng(0)
    n = 300

    def pairs(m, seed):
        r = np.random.default_rng(seed)
        x, y = r.integers(0, n, m), r.integers(0, n, m)
        k = x != y
        return np.unique(np.minimum(x, y)[k].astype(np.int64) * n + np.maximum(x, y)[k])

    def wstore(keys, fam):
        return DS.weighted_store(fam, n, keys // n, keys % n, rng.random(keys.size).astype(np.float32) + 0.1)

    def sstore(keys):
        g = SimpleNamespace(n_nodes=n, src=(keys // n).astype(np.int32), dst=(keys % n).astype(np.int32), weight=None,
                            rel=rng.integers(0, 3, keys.size).astype(np.int16))
        return FamilyStore.from_graph(g, "structural")

    class St:
        def __init__(self, d):
            self.d = d

        def get(self, f):
            return self.d[f]

    opened = 0
    for kb in (False, True):
        for cap in (3, 6, 1000):
            S, F0, T = pairs(3000, 1), pairs(1500, 2), pairs(400, 3)
            F0T = np.union1d(F0, T)
            KN = pairs(1200, 4)
            st_s = sstore(S)
            ner = wstore(F0T, "ner")
            if kb:
                ner, _ = DS.drop_pairs(ner, S)
            knn, _ = DS.drop_pairs(wstore(KN, "knn"), np.union1d(S, DS.pair_keys(ner)))
            keys = [S if kb else np.zeros(0, np.int64), np.zeros(0, np.int64) if kb else S, F0,
                    np.setdiff1d(F0T, F0), KN]
            need = {0: (0,) if kb else (1,), 1: (2, 3), 2: (4,)}
            stores = {"structural": st_s, "ner": ner, "knn": knn}
            qps, qed, eu, ev, ef, pool, raw = [], [], [], [], [], [], 0
            for _q in range(25):
                pl = rng.choice(n, size=int(rng.integers(20, 120)), replace=False).astype(np.int64)   # unsorted, as the look's
                edges, _t = MF.pool_edges(pl, stores, None, cap)
                m = 0
                for f, fam in enumerate(("structural", "ner", "knn")):
                    a, b, _attr = edges[fam]
                    eu.append(a), ev.append(b), ef.append(np.full(a.size, f, np.int64))
                    m += a.size
                raw += m
                qps.append(pl.size), qed.append(m), pool.append(pl)
            c = {"q_pool_size": np.asarray(qps), "q_edges": np.asarray(qed), "e_u": np.concatenate(eu),
                 "e_v": np.concatenate(ev), "e_fam": np.concatenate(ef), "pool": np.concatenate(pool)}
            T_ne, T_a, T_b, T_f, _ = DS.label_edges(c, NMAX)                       # zlink's family links
            L_ne, L_a, L_b, L_l, _ = DS.label_edges(c, NMAX, n, keys, need)       # D83's labelled links
            q, u, v, bits, fb, ob, nq, st = derive_part(L_ne, L_a, L_b, L_l, kb, cap)
            if kb:
                assert st["open_pairs"] == 0
            else:
                off = np.r_[0, np.cumsum(qps)]
                allp = np.concatenate(pool)
                fb, rs = resolve(q, u, v, bits, fb, ob, lambda i: allp[off[i]:off[i + 1]],
                                 St({0: st_s, 1: ner}), cap)
                st.update(rs)
                opened += st["open_pairs"]
            ne, a, b, f = emit(q, u, v, fb, nq)
            assert np.array_equal(ne, T_ne) and np.array_equal(a, T_a) and np.array_equal(b, T_b) and \
                np.array_equal(f, T_f.astype(np.int8)), (kb, cap, "D91 self-test: the links are not zlink's")
            assert messages(ne, a, f, cap) == raw, (kb, cap, "G1d does not hold on the look itself")
            if cap >= 1000:
                assert st["open_pairs"] == 0
            log(f"self-test kb={kb} cap={cap}: {st}")
    assert opened > 0, "the self-test decided no open pair"
    qn = np.array([5, 0, 7, 3, 0, 12, 1])
    bl = blocks(qn, 8)
    assert bl[0][0] == 0 and bl[-1][1] == qn.size and all(x[1] == y[0] for x, y in zip(bl, bl[1:]))
    assert all(x[3] - x[2] == int(qn[x[0]:x[1]].sum()) and (x[3] - x[2] <= 8 or x[1] - x[0] == 1) for x in bl), bl
    assert mine(("pools", "2wiki__fit.npz"), "2wiki") and mine(("cache", "2wiki", "fit", "part_0of3", "node.npy"), "2wiki")
    assert mine(("settings", "musique.json"), "musique") and mine(("cache", "basis_2wiki.npz"), "squad")
    assert not mine(("pools", "hotpotqa__fit.npz"), "2wiki") and not mine(("kt", "musique", "b1v", "kt.npy"), "2wiki")
    assert mine(("d83kt", "2wiki__b1v__0of1.npy"), "2wiki") and mine(("chains", "rmatch", "webqsp", "fit", "x"), "webqsp")
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="zd91_claim_"))
    try:
        assert set(OUT_OF) == set(ARMS) == set(CARVES) and all(OUT_OF[a].parent.name == "abl" and OUT_OF[a].name == a for a in ARMS)
        ab = tmp / "abl"
        claim("D91B", ab / "D91B")                                   # absent: claimed
        assert read_json(ab / "D91B" / CLAIM)["arm"] == "D91B" and claim("D91B", ab / "D91B")["arm"] == "D91B"
        for bad_arm, bad_dir in (("D91E", ab / "D91B"), ("D91B", tmp / "b1x"), ("D91B", tmp / "x" / "D91B")):
            try:
                claim(bad_arm, bad_dir)
                raise AssertionError(f"claim must refuse {bad_arm} on {bad_dir}")
            except SystemExit:
                pass
        ap2 = ab / "D91X"                                            # a folder of another stage: its own record and gate
        (ap2 / "records").mkdir(parents=True)
        (ap2 / "records" / "musique__b1x.json").write_text("{}", encoding="utf-8")
        (ap2 / "ap2_gate_musique.json").write_text("{}", encoding="utf-8")
        for arm_ in ("D91B",):
            try:
                CARVES["D91X"] = CARVES[arm_]
                ARMS["D91X"] = ARMS[arm_]
                claim("D91X", ap2)
                raise AssertionError("claim must refuse another stage's folder")
            except SystemExit:
                pass
            finally:
                CARVES.pop("D91X", None)
                ARMS.pop("D91X", None)
        own = ab / "D91F"                                            # D91's own folder before the claim existed: claimed
        (own / "records").mkdir(parents=True)
        (own / "records" / "squad__fit.json").write_text("{}", encoding="utf-8")
        (own / "links").mkdir()
        assert claim("D91F", own)["arm"] == "D91F"
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"zd91 selftest: OK ({opened} open pairs decided)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["arm", "gate", "survey", "selftest"])
    ap.add_argument("--arm")
    ap.add_argument("--dataset")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--dry", action="store_true", help="arm: derive, decide and check only; no file is written")
    ap.add_argument("--parts", type=int, help="arm --dry: the first N parts only")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "survey":
        if a.arm not in SRC_ARMS:
            raise SystemExit(f"survey: --arm is one of {sorted(SRC_ARMS)}")
        return 0 if survey(a.arm, a.dataset) else 1
    if a.arm not in ARMS:
        raise SystemExit(f"D91: --arm is one of {sorted(ARMS)}")
    if a.parts is not None and not a.dry:
        raise SystemExit("D91: --parts is for --dry runs only")
    if a.cmd == "arm":
        return arm_row(a.arm, a.dataset, a.host, a.dry, a.parts)
    return gate(a.arm, a.host)


if __name__ == "__main__":
    sys.exit(main())
