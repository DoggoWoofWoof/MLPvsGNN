"""D83 (docs/D83_UNIFIED_R_REFIT.md): the build stage of the one refit. The training ("fit") carves of the two WQ splits, and the
six s1eval carves, rebuilt on ABL's chain (zabl.py, not edited) with R's NER store = F0T, the S > R > K dedup and the edge
labels, on the POOLS OF RECORD (A2): each carve's pools file and KT rows are PKO's own (outputs/pko/pools), hard-linked into the
arm's root after a sha256 check, so the chain's pools and merge steps never run. Only what reads the stores is recomputed:
look, cache, rmatch and zrc (chained carves), links, typed (chained carves), kt (the pools' KT rows in the new cache's row
order) and the record.

    python outputs/mp_unified/zd83.py chain --arm D83F --dataset musique --lanes 3 --host
    python outputs/mp_unified/zd83.py gate --arm D83F --host
    python outputs/mp_unified/zd83.py link --arm D83E --dataset metaqa --host          (the pools of record, linked; also done by chain)
    python outputs/mp_unified/zd83.py selftest

Arms: D83F = the fit carves (squad, hotpotqa, 2wiki: fit; musique: s1fit; webqsp: fit), D83E = the six s1eval carves,
D83I = G1's identity arm (squad's fit carve and metaqa's s1eval carve with the record's stores, no dedup: section 8). The
structural slot is the record's on every dataset: R (U2b's W2_S) on the passage datasets, S (the KB store) on MetaQA and
WebQSP (section 1: the record's domain, where the pools of record exist). The NER store is F0T; the knn store is the package's.

    python outputs/mp_unified/zd83.py build --arm D83F --dataset musique --lanes 6 --host   (pools linked, look, bases, cache)

    python outputs/mp_unified/zd83.py chain --arm D83F --dataset musique --lanes 6 --host   (build, KT pass, then ABL's chain)
    python outputs/mp_unified/zd83.py g1 --host                                              (G1: D83I against the record)

The chain: the build (above), then the KT pass (the KT rows of the pools of record recomputed on the arm's stores, in
shards, `lanes` at a time, then merged and named in the arm's manifest), then zabl.chain, whose pools, merge, look, bases
and cache are filed, so it runs rmatch and zrc (chained carves), links (one link per label: labelled_build), typed (the
typed sets on the S links: chunk_typed_d83), kt and record. A D83F or D83E chain refuses until G1 has passed.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import d83stores as DS  # noqa: E402
import zabl as Z  # noqa: E402  (ABL: the chain, its roots and the gate; its source is not edited)

DECLARED = "docs/D83_UNIFIED_R_REFIT.md"
STAGE = "D83"
FIT_CARVE = {"squad": "fit", "hotpotqa": "fit", "2wiki": "fit", "musique": "s1fit", "webqsp": "fit"}   # zwq.FIT_CARVE
ARMS = {"D83I": {"squad": "fit", "metaqa": "s1eval"}, "D83F": FIT_CARVE, "D83E": {d: "s1eval" for d in Z.DATASETS}}
IDENTITY = "D83I"                                                  # G1: the record's stores through D83's plumbing
DEFERRED = ()                                                      # every step's D83 code is in (the KT pass, the labels)
LABELS = ("S", "R.mention", "R.name_comention", "R.title_comention", "K.knn")   # section 3, ids 0 to 4
SLOT = {d: ("S" if d in Z.KB else "R") for d in Z.DATASETS}                                       # the record's slot
U2R = ROOT / "outputs" / "u2r"
UL1 = ROOT / "outputs" / "ul1"
utc, sha_file, sha_src, rel, read_json, write_json = Z.utc, Z.sha_file, Z.sha_src, Z.rel, Z.read_json, Z.write_json
_ORIG_ON_ARM = Z.on_arm
_ORIG_POOLS = Z.pools


def log(msg):
    Z.log(f"zd83: {msg}")


def carve_of(arm, ds):
    if arm not in ARMS:
        raise SystemExit(f"D83: the arm is one of {', '.join(ARMS)}")
    if ds not in ARMS[arm]:
        raise SystemExit(f"D83: {arm} has no carve on {ds}")
    return ARMS[arm][ds]


def use(arm, ds):
    """zabl for this arm and dataset in this process: its carve, the record's slot, and this file as the program its chain starts."""
    Z.CARVE = carve_of(arm, ds)
    Z.ARM_ALL[arm] = (SLOT[ds], 1, 1)
    Z.__file__ = str(Path(__file__).resolve())
    Z.on_arm = _ORIG_ON_ARM if arm == IDENTITY else on_d83
    Z.pools = refuse_pools
    return Z.CARVE


def refuse_pools(*_a, **_k):
    raise SystemExit("D83: the pools are the record's (A2); the pools step never runs")


# ── F0T, R's NER store on a training corpus ─────────────────────────────────────

def f0t_pairs(d, n):
    """F0T's pairs and weights: CN_all (U2r) united with LT (UL1), LT's weight where both hold a pair. Both files are checked
    against their records (U2r's names.json, UL1's build.json) and UL1's node count."""
    nr = read_json(U2R / d / "names.json")
    if sha_file(U2R / d / "names.npz") != nr["names_npz_sha256"]:
        raise SystemExit(f"D83 G2: {d}: U2r's names.npz is not its record's")
    br = read_json(UL1 / d / "build.json")
    if sha_file(UL1 / d / "f0t.npz") != br["f0t_npz_sha256"] or int(br["nodes"]) != int(n) or int(nr["nodes"]) != int(n):
        raise SystemExit(f"D83 G2: {d}: UL1's f0t.npz is not its record's, or not of the store's {n} nodes")
    if br["names_npz_sha256"] != nr["names_npz_sha256"]:
        raise SystemExit(f"D83 G2: {d}: UL1 was built on other U2r names")
    with np.load(U2R / d / "names.npz") as z:
        f0 = (z["cn_all_src"], z["cn_all_dst"], z["cn_all_w"])
    with np.load(UL1 / d / "f0t.npz") as z:
        lt = (z["lt_src"], z["lt_dst"], z["lt_w"])
    a, b, w = DS.form_pairs([f0, lt], int(n))
    if a.size != int(br["f0t_pairs"]):
        raise SystemExit(f"D83 G2: {d}: F0T holds {a.size:,} pairs, UL1 filed {int(br['f0t_pairs']):,}")
    f0k = np.unique(f0[0].astype(np.int64) * int(n) + f0[1])
    return a, b, w, f0k


# ── the stores in the chain ──────────────────────────────────────────────────

@contextlib.contextmanager
def on_d83(M, name, arm):
    """zabl.on_arm with the record's slot, then the NER store replaced by F0T and the dedup applied (every patch put back on exit)."""
    with _ORIG_ON_ARM(M, name, arm) as A:
        mp_mod = M.S4.m3b_pools
        inner = mp_mod.load_or_build_store
        made = {}

        def structural_keys(ds_, cache_dir):
            if name in Z.PASSAGE:
                n = int(ds_.n_nodes)
                ip, col, _desc = M.U2C.graph_csr(name, "chosen", n)               # on_arm's slot (W2_S on a passage dataset)
                from mp_retrieval.m3b_pools import FamilyStore
                return DS.pair_keys(FamilyStore("structural", n, np.asarray(ip, np.int64), np.asarray(col, np.int32)))
            return DS.pair_keys(inner(ds_, "structural", cache_dir))

        def stores(ds_, cache_dir):
            if "ner" not in made:
                n = int(ds_.n_nodes)
                a, b, w, f0k = f0t_pairs(name, n)
                ner = ner0 = DS.weighted_store("ner", n, a, b, w)
                del a, b, w
                knn = inner(ds_, "knn", cache_dir)
                st_keys = structural_keys(ds_, cache_dir)
                if name in Z.KB:
                    ner, gone_s = DS.drop_pairs(ner, st_keys)
                else:
                    gone_s = 0
                knn2, gone_k = DS.drop_pairs(knn, np.union1d(st_keys, DS.pair_keys(ner)))
                write_label_sets(arm, name, n, label_sets(name, st_keys, f0k, DS.pair_keys(ner0), DS.pair_keys(knn)))
                made.update(ner=ner, knn=knn2, f0k=f0k, st_keys=st_keys,
                            counts={"structural_pairs": int(st_keys.size), "ner_pairs": int(ner.edges_stored),
                                    "ner_removed_in_S": int(gone_s), "knn_pairs_in": int(knn.edges_stored),
                                    "knn_removed_in_SR": int(gone_k), "knn_pairs": int(knn2.edges_stored)})
                log(f"{name}: stores {made['counts']}")
            return made

        def load_store(ds_, family, cache_dir):
            if ds_.name != name or family == "structural":
                return inner(ds_, family, cache_dir)
            if family in ("ner", "knn"):
                return stores(ds_, cache_dir)[family]
            raise SystemExit(f"D83: no store {family}")

        P_ = Z.B.Patches()
        try:
            P_.set(mp_mod, load_or_build_store=load_store)
            A.d83 = made
            yield A
        finally:
            P_.restore()


# ── the labels' key sets (section 3) ─────────────────────────────────────────

def label_sets(name, st_keys, f0k, f0t_keys, knn_keys):
    """The five labels' pair keys (min*n + max, sorted, unique) of one dataset, read from the stores BEFORE the dedup, so a
    pair keeps every label whose rule made it. On a KB the slot's pairs are S's; on a passage dataset they are W2_S's
    (R.mention). F0T's pairs that F0 lacks are R.title_comention's (on a KB, F0T before its pairs in S are removed)."""
    kb = name in Z.KB
    empty = np.zeros(0, np.int64)
    f0k = np.asarray(f0k, np.int64)
    st_keys = np.asarray(st_keys, np.int64)
    return {"S": st_keys if kb else empty, "R.mention": empty if kb else st_keys, "R.name_comention": f0k,
            "R.title_comention": np.setdiff1d(np.asarray(f0t_keys, np.int64), f0k),
            "K.knn": np.asarray(knn_keys, np.int64)}


def labels_dir(arm, name):
    return Z.arm_dir(arm) / "labels" / name


def write_label_sets(arm, name, n, sets):
    """Each label's keys as a .npy (np.save: the same bytes from any writer), written through a temporary name and
    os.replace; a set already on disk must equal this one. The record names each file's sha256 and the node count."""
    d = labels_dir(arm, name)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "dataset": name, "nodes": int(n), "labels": list(LABELS),
           "pairs": {}, "sha256": {}}
    for i, lab in enumerate(LABELS):
        k = np.ascontiguousarray(sets[lab], np.int64)
        if k.size and (not (np.diff(k) > 0).all() or int(k[0]) < 0 or int(k[-1]) >= int(n) * int(n)):
            raise SystemExit(f"D83: {name}: the {lab} keys are not sorted, unique and inside {n} nodes")
        f = d / f"{i}.npy"
        if f.exists():
            if not np.array_equal(np.load(f, allow_pickle=False), k):
                raise SystemExit(f"D83: {rel(f)} holds other {lab} keys than this build's")
        else:
            tmp = d / f"{i}.{os.getpid()}.tmp.npy"
            np.save(tmp, k)
            os.replace(tmp, f)
        rec["pairs"][lab] = int(k.size)
        rec["sha256"][lab] = sha_file(f)
    out = d / "record.json"
    if not out.exists():
        write_json(out, dict(rec, utc=utc()))
    elif {k_: v for k_, v in read_json(out).items() if k_ != "utc"} != rec:
        raise SystemExit(f"D83: {rel(out)} names other label sets than this build's")
    return rec


def fam_need(name):
    """The labels one of which an edge of each look family must carry (G4): the slot's (S on a KB, R.mention elsewhere),
    the NER store's (F0 or the title rule), the knn store's."""
    return {0: (0,) if name in Z.KB else (1,), 1: (2, 3), 2: (4,)}


def load_label_sets(arm, name):
    """The five key sets the look step wrote, each checked against the labels record. Returns (keys, nodes, record sha)."""
    d = labels_dir(arm, name)
    rec = read_json(d / "record.json")
    if rec["labels"] != list(LABELS) or rec["arm"] != arm or rec["dataset"] != name:
        raise SystemExit(f"D83: {rel(d)} is not this arm's label record")
    keys = []
    for i, lab in enumerate(LABELS):
        f = d / f"{i}.npy"
        if sha_file(f) != rec["sha256"][lab]:
            raise SystemExit(f"D83: {rel(f)} is not the file its record names")
        keys.append(np.load(f, allow_pickle=False))
    return keys, int(rec["nodes"]), sha_file(d / "record.json")


def d83_settings(ZL):
    return dict(ZL.SETTINGS, families=len(LABELS), labels=list(LABELS),
                edges="the look's pool edges, undirected (both directions), once per label the pair carries (D83), no self-loops")


def labelled_build(ds, carve, out_root, cache_root, look_root, placement=None, *, arm):
    """zlink.build with one link per (question, u, v, label) (d83stores.label_edges); on D83I the labels are the families and
    every array is zlink's own (G1). The record adds the label counts and the label sets' record."""
    import time
    import zlink as ZL
    LC, Rw = ZL.LC, ZL.R
    identity = arm == IDENTITY
    keys, n, lab_sha = (None, None, None) if identity else load_label_sets(arm, ds)
    need = None if identity else fam_need(ds)
    settings = ZL.SETTINGS if identity else d83_settings(ZL)
    L = 3 if identity else len(LABELS)
    t0 = time.time()
    d, files, _ids, head = LC.look_chunks(ds, carve, look_root)
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    summary = {}
    for p_, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"{p_}: built from another look than {d}")
        tp = time.time()
        lo, hi = r["chunks"]
        acc = {"q_ne": [], "e_u": [], "e_v": [], "e_fam": []}
        n_all, tot = [], {"raw": 0, "self_loops": 0, "kept": 0}
        labs = np.zeros(L, np.int64)
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                c = {k: zf[k] for k in ("q_pool_size", "q_edges", "e_u", "e_v", "e_fam") + (() if identity else ("pool",))}
            ne, a, b, ff, st = DS.label_edges(c, ZL.NMAX, n, keys, need, str(files[ch]))
            for k, x in zip(("q_ne", "e_u", "e_v", "e_fam"), (ne, a, b, ff)):
                acc[k].append(x)
            for k in tot:
                tot[k] += st[k]
            labs += np.asarray(st["labels"], np.int64)
            n_all.append(c["q_pool_size"].astype(np.int64))
        arrays = {k: np.concatenate(v) for k, v in acc.items()}
        n_all = np.concatenate(n_all)
        if n_all.size != r["queries"] or int(n_all.sum()) != r["rows"]:
            raise SystemExit(f"{p_}: {n_all.size} queries and {int(n_all.sum())} rows against {r['queries']} and {r['rows']}")
        if (p_ / "n.npy").exists():
            if not np.array_equal(np.asarray(LC.load_array(p_, "n", r, verify=True)).astype(np.int64), n_all):
                raise SystemExit(f"{p_}: the chunks' pool sizes are not step 1's cached n")
            against = "IDENTICAL"
        else:
            against = "step 1's arrays are not on disk here"
        out = Path(out_root) / ds / carve / p_.name
        out.mkdir(parents=True, exist_ok=True)
        shas = {k: Rw.save_npy(out, k, v) for k, v in arrays.items()}
        fam = np.bincount(arrays["e_fam"].astype(np.int64), minlength=L)
        if not np.array_equal(fam, labs):
            raise SystemExit(f"{p_}: the per-label counts do not add up")
        rec = {"dataset": ds, "carve": carve, "part": p_.name, "step1_record_sha256": LC.sha_file(p_ / "record.json"),
               "chunks": [lo, hi], "queries": int(n_all.size), "rows": r["rows"], "edges": int(arrays["e_u"].size),
               "edges_per_row": round(arrays["e_u"].size / max(1, r["rows"]), 3), "families": [int(x) for x in fam],
               "look_edges": tot, "look": head["records"], "carve_ids_sha256": head["carve_ids_sha256"],
               "n_against_step1": against, "settings": settings,
               "arrays": {k: {"dtype": v.dtype.str, "shape": list(v.shape), "sha256": shas[k]} for k, v in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "seconds": round(time.time() - tp, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if not identity:
            rec["d83"] = {"declared_in": DECLARED, "arm": arm, "labels": dict(zip(LABELS, (int(x) for x in fam))),
                          "label_sets_record_sha256": lab_sha, "family_label_misses": 0}
        Rw.write_json(out / "record.json", rec)
        summary[p_.name] = {"queries": int(n_all.size), "edges": rec["edges"], "against": against}
        log(f"links {ds}/{carve} {p_.name}: {n_all.size} queries, {rec['edges']} links ({rec['edges_per_row']} a row; "
            f"{'families' if identity else 'labels'} {rec['families']}) ({time.time() - t0:.0f}s)")
    return summary


def chunk_typed_d83(c, where, n_rel, k_rel, *, orig, keys, n, need, base_keys):
    """zu2n.chunk_typed on labelled links: the typed sets and direction bits zu2n gives each structural (family 0) link go
    to the S link of the same (question, u, v); links of the other labels carry none. Returns zu2n's tuple on the
    labelled links."""
    import zlink as ZL
    NMAX = ZL.NMAX
    (ne0, a0, b0, f0), ue0, pad0, tdir0, st0 = orig({k: c[k] for k in base_keys}, where, n_rel, k_rel)
    ne, a, b, lab, _cnt = DS.label_edges(c, NMAX, n, keys, need, where)
    E0, E = int(a0.size), int(a.size)
    q0 = np.repeat(np.arange(ne0.size), ne0)
    q1 = np.repeat(np.arange(ne.size), ne)
    k0 = (q0 * NMAX + a0.astype(np.int64)) * NMAX + b0.astype(np.int64)
    k1 = (q1 * NMAX + a.astype(np.int64)) * NMAX + b.astype(np.int64)
    i0 = np.nonzero(np.asarray(f0) == 0)[0]
    i1 = np.nonzero(np.asarray(lab) == need[0][0])[0]                  # the slot's label (S on a KB: typed runs on KBs)
    if i0.size != i1.size or not np.array_equal(k0[i0], k1[i1]):
        raise SystemExit(f"{where}: G4: the S links are not the look's structural links ({i1.size} against {i0.size})")
    srow = np.full(E0, -1, np.int64)
    srow[ue0] = np.arange(ue0.size)
    sr = srow[i0]
    ue = i1[sr >= 0]
    pad = pad0[sr[sr >= 0]]
    tdir = np.zeros(E, np.int8)
    tdir[i1] = tdir0[i0]
    st = dict(st0, edges=E, family0=int(i1.size), typed=int(ue.size), empty_family0=int(i1.size - ue.size))
    return (ne, a, b, lab), ue, pad, tdir, st


@contextlib.contextmanager
def step_patches(kind, arm, name):
    """The links step builds labelled links; the typed step puts zu2n's sets on the S links (D83 arms only; on D83I zu2n
    runs unpatched on the family links, which G1 compares with the record's)."""
    P_ = Z.B.Patches()
    try:
        if kind == "links":
            import functools
            import zlink as ZL
            P_.set(ZL, build=functools.partial(labelled_build, arm=arm))
        elif kind == "typed" and arm != IDENTITY:
            import functools
            import zlink as ZL
            import zu2n as U2N
            keys, n, _sha = load_label_sets(arm, name)
            P_.set(U2N, chunk_typed=functools.partial(chunk_typed_d83, orig=U2N.chunk_typed, keys=keys, n=n,
                                                      need=fam_need(name), base_keys=tuple(U2N.TKEYS)),
                   TKEYS=tuple(U2N.TKEYS) + ("pool",))
            P_.set(ZL, SETTINGS=d83_settings(ZL))
        yield
    finally:
        P_.restore()


# ── the KT pass: the pools' KT rows on the arm's stores ──────────────────────

def kt_dir(R):
    return Path(R.base) / "d83kt"


def kt_shards(name):
    return 2 * int(Z.SHARDS[name])


def kt_pass(arm, name, shard, host=False):
    """One shard's KT rows: for each question, P1c3's columns (zpko.rule_batch's KT call) on the arm's stores, at the
    question's filed pool (B and the kept expansion) of the pools of record. Writes <stem>__<i>of<k>.npy and .json."""
    import time
    M = Z.B.mods(host)
    PK = M.PK
    R = Z.roots_of(M, arm)
    carve = Z.CARVE
    i, k = shard
    st = PK.stem(name, carve)
    out = kt_dir(R) / f"{st}__{i}of{k}"
    if out.with_suffix(".json").exists():
        log(f"KT pass {arm}/{name} {i}/{k}: filed")
        return 0
    t0 = time.time()
    filed, ent, msha = PK.load_filed(name, carve, R)
    bp, bn = np.asarray(filed["bs_ptr"], np.int64), np.asarray(filed["bs_nodes"], np.int64)
    ep, en = np.asarray(filed["ptr"], np.int64), np.asarray(filed["nodes"], np.int64)
    ent_flag = np.asarray(filed["entity"], bool)
    with Z.B.checks(M), Z.on_arm(M, name, arm):
        cx, ctx = PK.open_pko(name, carve, 2)
        t_open = time.time() - t0
        ids = PK.question_ids(cx)
        if ids != [str(x) for x in filed["ids"]]:
            raise SystemExit(f"D83 KT pass {name}/{carve}: the carve's questions are not the pools file's")
        nq = len(ids)
        lo, hi = nq * i // k, nq * (i + 1) // k
        K, D, prep, constant, threads = cx.K, cx.D, cx.prep, cx.constant, cx.threads
        rows, changed = [], 0
        for b0 in range(lo, hi, PK.Q.BATCH):
            js = list(range(b0, min(hi, b0 + PK.Q.BATCH)))
            srcs = D.sources_batch(prep, js, cx.linked, cx.unions, constant, threads)
            steered = K.steered_batch(prep, js, cx.linked, cx.unions, srcs, constant, threads)
            for t, j in enumerate(js):
                src, _nn, (L_, Ls_) = srcs[t]
                route = D.routes(L_, Ls_)
                knd = K.kind(Ls_, prep.seeds[j], cx.nonly)
                exp = en[ep[j]:ep[j + 1]]
                pool = np.union1d(bn[bp[j]:bp[j + 1]], exp)
                oe, pe = K.p1e_orders(src, steered[t], bool(route), knd, cx.n_st, constant)
                ktc = PK.KT3.columns(PK.ZP.kt_orders(oe, pe), pool, exp, knd == "entity").astype(PK.ZP.KT_DT)
                if ktc.shape[0] != pool.size:
                    raise SystemExit(f"D83 KT pass {name}/{carve} question {j}: one KT row per pool row is not what P1c3 gave")
                changed += int((knd == "entity") != bool(ent_flag[j]))
                rows.append(ktc)
            log(f"KT pass {arm}/{name} {i}/{k}: {b0 + len(js) - lo}/{hi - lo} questions, {time.time() - t0:.0f} s")
    kt = np.concatenate(rows) if rows else np.zeros((0, PK.KT3.NKT), PK.ZP.KT_DT)
    want = int((np.diff(bp)[lo:hi] + np.diff(ep)[lo:hi]).sum())
    if kt.shape[0] != want:
        raise SystemExit(f"D83 KT pass {name}/{carve} {i}/{k}: {kt.shape[0]} rows against the pools' {want}")
    out.parent.mkdir(parents=True, exist_ok=True)
    sha = PK.ZP.save_npy(out.with_suffix(".npy"), kt)
    write_json(out.with_suffix(".json"), {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "dataset": name, "carve": carve,
                                          "shard": [i, k], "range": [lo, hi], "rows": int(kt.shape[0]), "sha256": sha,
                                          "kind_changed": changed, "pools_manifest_sha256": msha,
                                          "pools_file_sha256": ent["sha256"], "seconds": {"open": round(t_open, 1),
                                          "pass": round(time.time() - t0 - t_open, 1)}, "zd83_sha256": sha_src(__file__),
                                          "utc": utc()})
    log(f"KT pass {arm}/{name} {i}/{k}: {hi - lo} questions, {kt.shape[0]} rows, kind changed {changed} "
        f"({time.time() - t0:.0f} s)")
    return 0


def kt_merge(arm, name, host=False):
    """The shards' KT rows joined in question order, saved beside the linked pools and named in the arm's manifest (its
    kt block); the record's KT file and sha256 stay in the manifest's d83 block. On D83I they must be the record's."""
    M = Z.B.mods(host)
    PK = M.PK
    R = Z.roots_of(M, arm)
    carve = Z.CARVE
    st = PK.stem(name, carve)
    man = R.pools / f"{st}.json"
    ent = read_json(man)
    if ent.get("d83", {}).get("kt_recomputed"):
        return 0
    k = kt_shards(name)
    parts, recs = [], []
    for i in range(k):
        base = kt_dir(R) / f"{st}__{i}of{k}"
        r = read_json(base.with_suffix(".json"))
        if sha_file(base.with_suffix(".npy")) != r["sha256"]:
            raise SystemExit(f"D83: {rel(base)}.npy is not the file its record names")
        parts.append(np.load(base.with_suffix(".npy"), allow_pickle=False))
        recs.append(r)
    kt = np.concatenate(parts)
    filed, _ent, _msha = PK.load_filed(name, carve, R)
    PK.check_filed(filed, kt)
    rec_kt = PK.resolve(ent["kt"]["file"])
    same = bool(np.array_equal(np.load(rec_kt, allow_pickle=False), kt))
    if arm == IDENTITY and not same:
        raise SystemExit(f"D83 G1: {name}/{carve}: the KT pass on the record's stores is not the record's KT rows")
    f = R.pools / f"{st}.d83kt.npy"
    sha = PK.ZP.save_npy(f, kt)
    d83 = dict(ent["d83"], kt_recomputed=True, record_kt={"file": ent["kt"]["file"], "sha256": ent["kt"]["sha256"]},
               kt_equal_to_record=same, kind_changed=int(sum(r["kind_changed"] for r in recs)),
               kt_shards=[sha_file((kt_dir(R) / f"{st}__{i}of{k}").with_suffix(".json")) for i in range(k)], kt_utc=utc())
    write_json(man, dict(ent, kt={"file": rel(f), "sha256": sha, "shape": list(kt.shape)}, d83=d83))
    log(f"KT merge {arm}/{name}/{carve}: {kt.shape[0]} rows; equal to the record's: {same}; kind changed {d83['kind_changed']}")
    return 0


def g1_passed():
    p = Z.arm_dir(IDENTITY) / "d83_g1.json"
    return p.exists() and read_json(p).get("verdict") == "PASS"


def chain(arm, name, lanes=1, host=False):
    """Build, the KT pass (shards `lanes` at a time, then the merge), then zabl.chain for the rest."""
    if arm != IDENTITY and not g1_passed():
        raise SystemExit(f"D83 chain {arm}/{name}: G1 has not passed (outputs/abl/{IDENTITY}/d83_g1.json)")
    build(arm, name, lanes, host)
    import concurrent.futures as cf
    import subprocess
    M = Z.B.mods(host)
    R = Z.roots_of(M, arm)
    st = M.PK.stem(name, Z.CARVE)
    if not read_json(R.pools / f"{st}.json").get("d83", {}).get("kt_recomputed"):
        base = ([sys.executable, str(Z.PYLIB_RUN)] if host else [sys.executable]) + [str(Path(__file__).resolve())]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
        k = kt_shards(name)

        def one(i):
            cmd = base + ["ktpass", "--arm", arm, "--dataset", name, "--shard", f"{i}/{k}"] + (["--host"] if host else [])
            if subprocess.run(cmd, cwd=str(ROOT), env=env, check=False).returncode:
                raise SystemExit(f"D83 chain {arm}/{name}: KT pass {i}/{k} failed")

        todo = [i for i in range(k) if not (kt_dir(R) / f"{st}__{i}of{k}.json").exists()]
        with cf.ThreadPoolExecutor(max_workers=max(1, int(lanes))) as ex:
            for fut in [ex.submit(one, i) for i in todo]:
                fut.result()
        kt_merge(arm, name, host)
    return Z.chain(arm, name, lanes, host)


# ── G1: the identity arm against the record ──────────────────────────────────

def g1(host=False):
    """Every D83I array named by a record (cache parts, links, typed, kt) equal to the record's by sha256, the look's chunk
    arrays equal to the record's, the KT pass's rows the record's (kt_merge) and the linked pools the record's."""
    M = Z.B.mods(host)
    PK = M.PK
    src, R = PK.roots(), Z.roots_of(M, IDENTITY)
    out = {"declared_in": DECLARED, "stage": STAGE, "gate": "G1", "carves": {}, "zd83_sha256": sha_src(__file__)}
    bad = []
    for name, carve in ARMS[IDENTITY].items():
        c = out["carves"][f"{name}/{carve}"] = {}
        man = read_json(R.pools / f"{PK.stem(name, carve)}.json")
        c["kt_pass_equal"] = bool(man.get("d83", {}).get("kt_equal_to_record"))
        if not c["kt_pass_equal"]:
            bad.append(f"{name}: KT pass")
        for what in ("cache", "links", "typed", "kt"):
            a, b = Path(getattr(src, what)) / name / carve, Path(getattr(R, what)) / name / carve
            ra = sorted(a.rglob("record.json")) if a.exists() else []
            if what == "typed" and not ra:
                c[what] = "none in the record"
                continue
            n_eq = n_all = 0
            for r_ in ra:
                rb = b / r_.relative_to(a)
                ea, eb = read_json(r_), (read_json(rb) if rb.exists() else {})
                arrs = ea.get("arrays", {})
                for key, v in arrs.items():
                    n_all += 1
                    if eb.get("arrays", {}).get(key, {}).get("sha256") == v.get("sha256"):
                        n_eq += 1
                    else:
                        bad.append(f"{name}/{what}/{r_.parent.name}/{key}")
            c[what] = {"records": len(ra), "arrays": n_all, "equal": n_eq}
            if not ra or n_all == 0:
                bad.append(f"{name}/{what}: no record to compare")
        la, lb = Path(src.look) / name / carve / "chunks", Path(R.look) / name / carve / "chunks"
        fa = sorted(la.glob("c*.npz"))
        n_eq = 0
        for f in fa:
            g = lb / f.name
            if not g.exists():
                bad.append(f"{name}/look/{f.name} missing")
                continue
            with np.load(f) as za, np.load(g) as zb:
                if set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files):
                    n_eq += 1
                else:
                    bad.append(f"{name}/look/{f.name}")
        c["look_chunks"] = {"record": len(fa), "equal": n_eq}
        if not fa:
            bad.append(f"{name}/look: the record's chunks are not on disk")
    out.update(verdict="FAIL" if bad else "PASS", differ=bad[:50], n_differ=len(bad), utc=utc())
    write_json(Z.arm_dir(IDENTITY) / "d83_g1.json", out)
    log(f"G1: {out['verdict']} ({len(bad)} differ{': ' + ', '.join(bad[:6]) if bad else ''})")
    return 0 if not bad else 1


# ── the pools of record in the arm's root ────────────────────────────────────

def link_pools(arm, ds, host=False):
    """PKO's pools file and KT rows of the carve hard-linked into the arm's pools root (zwq.place: sha256 checked, nothing
    overwritten) and its manifest written there naming them, with a d83 block. The chain then skips pools and merge."""
    M = Z.B.mods(host)
    PK = M.PK
    carve = carve_of(arm, ds)
    src_R, R = PK.roots(), Z.roots_of(M, arm)
    st = PK.stem(ds, carve)
    man_src = src_R.pools / f"{st}.json"
    ent = read_json(man_src)
    out = R.pools / f"{st}.json"
    import zwq as WQ
    files = {"file": (PK.resolve(ent["file"]), ent["sha256"]), "kt": (PK.resolve(ent["kt"]["file"]), ent["kt"]["sha256"])}
    files = {k: (f, R.pools / Path(f).name, sha) for k, (f, sha) in files.items()}
    placed = {k: WQ.place(f, dst, sha) for k, (f, dst, sha) in files.items()}
    new = json.loads(json.dumps(ent))
    new["file"] = rel(files["file"][1])
    new["kt"] = dict(ent["kt"], file=rel(files["kt"][1]))
    new["d83"] = {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "linked_from": rel(man_src),
                  "record_manifest_sha256": sha_file(man_src), "placed": placed, "utc": utc()}
    if out.exists():
        old = read_json(out)
        old_kt = old.get("d83", {}).get("record_kt", old.get("kt", {}))
        if old.get("sha256") != ent["sha256"] or old_kt.get("sha256") != ent["kt"]["sha256"]:
            raise SystemExit(f"D83: {rel(out)} is present and names other pools than the record's")
        return 0
    write_json(out, new)
    log(f"link {arm}/{ds}/{carve}: the record's pools {placed}")
    return 0


def build(arm, name, lanes=1, host=False):
    """The pools of record linked, then zabl.chain's first steps, each its own process: the look shards, the bases and the
    cache parts (`lanes` at a time). A step already filed is skipped. The later steps wait on their code (DEFERRED)."""
    link_pools(arm, name, host)
    if host:
        log(f"build {arm}/{name}: below-normal priority {'set' if Z.lower_priority() else 'not set'}")
    import concurrent.futures as cf
    import subprocess
    import time
    M = Z.B.mods(host)
    PK = M.PK
    R = Z.roots_of(M, arm)
    base = ([sys.executable, str(Z.PYLIB_RUN)] if host else [sys.executable]) + [str(Path(__file__).resolve())]
    tail = ["--host"] if host else []
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
    t0 = time.time()

    def one(kind, flag=None):
        cmd = base + ["step", kind, "--arm", arm, "--dataset", name] + (flag or []) + tail
        t = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False).returncode
        if rc:
            raise SystemExit(f"D83 build {arm}/{name}: {kind} {flag or ''} exited {rc}")
        log(f"build {arm}/{name}: {kind} {flag or ''} done in {time.time() - t:.0f} s")

    def many(kind, flags):
        if flags:
            with cf.ThreadPoolExecutor(max_workers=max(1, int(lanes))) as ex:
                for fut in [ex.submit(one, kind, f) for f in flags]:
                    fut.result()

    carve = Z.CARVE
    ent = read_json(R.pools / f"{PK.stem(name, carve)}.json")
    nl, nc = PK.look_parts(name, int(ent["questions"]), int(ent["rows"]))
    bad = PK.other_counts(R, name, carve, nl, nc)
    if bad:
        raise SystemExit(f"D83 build {arm}/{name}: files of other shard or part counts: {[rel(p) for p in bad[:4]]}")
    todo = [i for i in range(nl) if not all(p.exists() for p in PK.look_outs(R, name, carve, (i, nl)))]
    if todo:
        Z.clean_look(R, name, nl)
    many("look", [["--index", f"{i}/{nl}"] for i in todo])
    if not all((Path(R.cache) / f"basis_{b}.{x}").exists() for b in ("2wiki", "hotpotqa") for x in ("npz", "json")):
        one("bases")
    many("cache", [["--index", f"{i}/{nc}"] for i in range(nc)
                   if not (Path(R.cache) / name / carve / f"part_{i}of{nc}" / "record.json").exists()])
    lab = labels_dir(arm, name) / "record.json"
    write_json(Path(R.base) / "steps" / f"{name}__{carve}__d83build.json",
               {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "dataset": name, "carve": carve, "look_shards": nl,
                "cache_parts": nc, "questions": int(ent["questions"]), "rows": int(ent["rows"]),
                "labels_record_sha256": sha_file(lab) if lab.exists() else None, "zd83_sha256": sha_src(__file__),
                "seconds": round(time.time() - t0, 1), "utc": utc()})
    log(f"build {arm}/{name}/{carve}: look ({nl} shards), bases and cache ({nc} parts) filed ({time.time() - t0:.0f} s)")
    return 0


def built(arm):
    return tuple(ds for ds in ARMS[arm] if (Z.arm_dir(arm) / "records" / f"{ds}__{ARMS[arm][ds]}.json").exists())


def gate(arm, datasets=None, host=False):
    M = Z.B.mods(host)
    R = Z.roots_of(M, arm)
    datasets = tuple(datasets or built(arm))
    if not datasets:
        raise SystemExit(f"D83: no carve of {arm} is filed")
    with Z.B.checks(M):
        verdict = M.PK.gate(R, carves=[(ds, ARMS[arm][ds]) for ds in datasets])
    write_json(Z.arm_dir(arm) / f"d83_gate_{'-'.join(datasets)}.json",
               {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "pko_gate": rel(R.gate), "pko_gate_sha256": sha_file(R.gate),
                "verdict": verdict, "carves": [f"{ds}/{ARMS[arm][ds]}" for ds in datasets], "zd83_sha256": sha_src(__file__),
                "utc": utc()})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("--arm")
    ap.add_argument("--dataset")
    ap.add_argument("--datasets")
    ap.add_argument("--index")
    ap.add_argument("--shard")
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.arm not in ARMS:
        raise SystemExit(f"D83: --arm is one of {', '.join(ARMS)}")
    if a.cmd == "g1":
        return g1(a.host)
    if a.cmd in ("chain", "step", "link", "build", "ktpass"):
        if not a.dataset:
            raise SystemExit(f"{a.cmd}: --dataset is required")
        use(a.arm, a.dataset)
    if a.cmd == "link":
        return link_pools(a.arm, a.dataset, a.host)
    if a.cmd == "build":
        return build(a.arm, a.dataset, a.lanes, a.host)
    if a.cmd == "chain":
        return chain(a.arm, a.dataset, a.lanes, a.host)
    if a.cmd == "ktpass":
        return kt_pass(a.arm, a.dataset, Z.parse_pair(a.shard), a.host)
    if a.cmd == "step":
        if a.kind in ("pools", "merge"):
            raise SystemExit("D83: the pools are the record's (A2); pools and merge never run")
        if a.kind == "kt" and not read_json(Z.roots_of(Z.B.mods(a.host), a.arm).pools /
                                            f"{a.dataset}__{Z.CARVE}.json").get("d83", {}).get("kt_recomputed"):
            raise SystemExit("D83: the kt step reads the KT pass's rows; run the KT pass first")
        with step_patches(a.kind, a.arm, a.dataset):
            return Z.run_step(a.kind, a.arm, a.dataset, Z.parse_pair(a.index), Z.parse_pair(a.shard), a.host)
    if a.cmd == "gate":
        return gate(a.arm, tuple(a.datasets.split(",")) if a.datasets else None, a.host)
    raise SystemExit(__doc__)


def selftest():
    DS.selftest()
    assert set(FIT_CARVE) == {"squad", "hotpotqa", "2wiki", "musique", "webqsp"} and set(ARMS["D83E"]) == set(Z.DATASETS)
    import zwq
    assert FIT_CARVE == zwq.FIT_CARVE, (FIT_CARVE, zwq.FIT_CARVE)
    assert SLOT == {"musique": "R", "squad": "R", "hotpotqa": "R", "2wiki": "R", "metaqa": "S", "webqsp": "S"}
    keep = (Z.CARVE, Z.__file__, dict(Z.ARM_ALL), Z.on_arm, Z.pools)
    try:
        assert use("D83F", "musique") == "s1fit" and Z.ARM_ALL["D83F"] == ("R", 1, 1) and Z.on_arm is on_d83
        assert use("D83E", "webqsp") == "s1eval" and Z.ARM_ALL["D83E"] == ("S", 1, 1) and Z.dropped("D83E") == ()
        assert Z.arm_dir("D83F") == Z.OUT / "D83F" and Z.arm_dir("D83E") == Z.OUT / "D83E"
        assert use("D83I", "squad") == "fit" and Z.on_arm is _ORIG_ON_ARM and use("D83I", "metaqa") == "s1eval"
        refused = [lambda: use("D83F", "metaqa"), lambda: use("RK", "musique"), lambda: use("D83I", "musique"),
                   lambda: Z.pools("D83F", "musique"), lambda: main(["step", "pools", "--arm", "D83F", "--dataset", "squad"])]
        if not g1_passed():
            refused.append(lambda: main(["chain", "--arm", "D83F", "--dataset", "squad"]))
        for bad in refused:
            try:
                bad()
                raise AssertionError("must refuse")
            except SystemExit:
                pass
    finally:
        Z.CARVE, Z.__file__, Z.on_arm, Z.pools = keep[0], keep[1], keep[3], keep[4]
        Z.ARM_ALL.clear()
        Z.ARM_ALL.update(keep[2])
    # the label sets: from the stores before the dedup; R.title_comention = F0T minus F0; S on a KB, R.mention elsewhere
    st_k, f0, f0t, kn = (np.array(x, np.int64) for x in ([3, 9, 14], [5, 9, 20], [5, 9, 11, 20, 30], [3, 7, 11]))
    ls = label_sets("musique", st_k, f0, f0t, kn)
    assert ls["S"].size == 0 and ls["R.mention"].tolist() == [3, 9, 14] and ls["R.title_comention"].tolist() == [11, 30]
    lk = label_sets("metaqa", st_k, f0, f0t, kn)
    assert lk["S"].tolist() == [3, 9, 14] and lk["R.mention"].size == 0 and lk["K.knn"].tolist() == [3, 7, 11]
    import tempfile
    keep_out = Z.OUT
    with tempfile.TemporaryDirectory() as td:
        try:
            Z.OUT = Path(td)
            r1 = write_label_sets("D83F", "musique", 8, ls)
            r2 = write_label_sets("D83F", "musique", 8, ls)                   # a second writer: the same files
            assert r1 == r2 and r1["pairs"]["R.title_comention"] == 2
            assert np.load(labels_dir("D83F", "musique") / "1.npy").tolist() == [3, 9, 14]
            try:
                write_label_sets("D83F", "musique", 8, dict(ls, **{"K.knn": np.array([1, 2], np.int64)}))
                raise AssertionError("other keys must be refused")
            except SystemExit:
                pass
        finally:
            Z.OUT = keep_out
    # chunk_typed_d83 on a toy KB chunk: the S links take zu2n's sets of the structural links; other labels carry none
    import zlink as ZL
    import zu2n as U2N
    ck = {"q_pool_size": np.array([3, 3]), "q_edges": np.array([3, 2]), "e_u": np.array([0, 1, 0, 0, 2]),
          "e_v": np.array([1, 2, 2, 1, 1]), "e_fam": np.array([0, 0, 2, 0, 1]), "e_fwd": np.array([1, 0, 0, 1, 0]),
          "e_bwd": np.array([0, 1, 0, 0, 0]), "e_rel": np.array([[2, -1], [3, 1], [-1, -1], [0, -1], [-1, -1]]),
          "pool": np.array([0, 1, 2, 5, 6, 7])}
    nn = 10
    kk = lambda pairs: np.unique(np.array([min(p_) * nn + max(p_) for p_ in pairs], np.int64))  # noqa: E731
    keys = [kk([(0, 1), (1, 2), (5, 6)]), np.zeros(0, np.int64), kk([(6, 7), (0, 1)]), np.zeros(0, np.int64), kk([(0, 2)])]
    base_keys = tuple(U2N.TKEYS)
    (ne, a, b, lab), ue, pad, tdir, st = chunk_typed_d83(ck, "toy", 4, 2, orig=U2N.chunk_typed, keys=keys, n=nn,
                                                         need=fam_need("metaqa"), base_keys=base_keys)
    (ne0, a0, b0, f0), ue0, pad0, tdir0, _st0 = U2N.chunk_typed({k: ck[k] for k in base_keys}, "toy", 4, 2)
    s_idx = np.nonzero(lab == 0)[0]
    assert np.array_equal(ue, s_idx[np.isin(np.nonzero(f0 == 0)[0], ue0)])
    assert np.array_equal(pad, pad0) and np.array_equal(tdir[s_idx], tdir0[f0 == 0]) and not tdir[lab != 0].any()
    q_of = np.repeat(np.arange(ne.size), ne)
    triples = {(int(q), int(x), int(y), int(z)) for q, x, y, z in zip(q_of, a, b, lab)}
    assert (0, 0, 1, 2) in triples and (0, 0, 1, 0) in triples and (1, 0, 1, 0) in triples and st["typed"] == ue.size
    assert ZL.SETTINGS["families"] == 3 and d83_settings(ZL)["families"] == 5
    with step_patches("links", "D83F", "musique"):
        assert ZL.build.func is labelled_build and ZL.build.keywords == {"arm": "D83F"}
    assert ZL.build is not labelled_build and getattr(ZL.build, "func", None) is None
    assert fam_need("metaqa")[0] == (0,) and fam_need("2wiki")[0] == (1,) and kt_shards("musique") == 6
    print("zd83 self-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
