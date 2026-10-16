"""T1 (docs/T1_GRANULARITY_TRANSFORMS.md): S6's granularity transforms as a training augmentation of the lean pair.
S6 (chainscore19.py) trained its chain scorer on metaqa with the relation layer rewritten, and the rewrite carried to
webqsp zero-shot (em-gr +6.43 R@5). The lean pair's chain input (rmatch's and zrc's entries, read by zrckp's and
zspk's chain heads) is learnt on metaqa alone, at metaqa's own granularity (9 relation names, no mediators), and read
zero-shot on webqsp, whose relations are many and whose facts run through CVT mediators.

Only metaqa's fit carve changes, and only in training. Each of its questions takes one regime, fixed by a hash of its
id (so its weight in training is unchanged):
    id    the question as filed
    mg3   chainscore19's mg3: relation r -> perm(r) // 3, the merged relation named by its members' mean name
    rf    chainscore19's rf on this question: a hashed half of the relations reified, (h, r, t) -> (h, r, m) and
          (m, r', t), one mediator m per distinct (h, r, t) beyond the pool's rows, r' named as r; a mediator is never
          a row, so its entries are dropped and the walk only passes through it
The relation table is one union [the 9 as filed | the 3 merged | the 9 second halves], and its centred form is centred
on the filed 9's mean, so an id question's entries, tables and chain features are today's bit for bit. Reads,
serving, every other carve and every parameter are the arm's own.

    python outputs/mp_unified/zt1gr.py build --host
    python outputs/mp_unified/zt1gr.py train --split L-musique --name scr-t1-zrckpt --arm zrckpt --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zt1gr.py read --name scr-t1-zrckpt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zt1gr.py compare --new outputs/screen/fits/scr-t1-zrckpt \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-t1-zrckpt
    python outputs/mp_unified/zt1gr.py pair --arm zrckpt --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zt1gr.py recall --arm zrckpt --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zt1gr.py verdict
    python outputs/mp_unified/zt1gr.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3p as Z3P  # noqa: E402  (zrckp; through it P1c3's zspk and KT carves, P1c2's roots, gate and card)
import chainscore19 as CS19  # noqa: E402  (S6's transforms: mg3's permutation; the selftest checks rf against it)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z3 = Z3P.Z3
Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/T1_GRANULARITY_TRANSFORMS.md"
OUT = ROOT / "outputs" / "t1"
SCREEN = ROOT / "outputs" / "screen"
DS = "metaqa"
CARVE = LG.FIT_CARVE[DS]
REGIMES = ("id", "mg3", "rf")
MG = 3                                                    # mg3 merges three relations at a time
MG_SEED = CS19.SEED + 3                                   # chainscore19's mg3 permutation
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpt": "zrckp", "zspkt": "zspk"}
ARM_NAMES = tuple(BASE_OF)
ARMS = {"zrckpt": Z3P.ARMS["zrckp"], "zspkt": Z3.ARMS["zspk"]}
S.ARMS.update(ARMS)
FIT = {a: {"L-musique": f"scr-t1-{a}", "L-hotpotqa": f"scr-t1-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpt": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkt": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}), metaqa as filed"
           for a in ARM_NAMES}
PAIR = {a: SCREEN / f"scr-t1-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-t1-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not T1's, or not twelve reads "
        "(docs/T1_GRANULARITY_TRANSFORMS.md, section 3)")
LOADED = []                                               # the carves read under T1's chains in this process


def roots(out=OUT):
    return {"rmatch": Path(out) / "chains" / "rmatch", "zrc": Path(out) / "chains" / "zrc",
            "rel": Path(out) / "relations", "build": Path(out) / "build.json"}


# ── the transforms ───────────────────────────────────────────────────────────


def layout(n0):
    """mg3's permutation (chainscore19's), the merged count and the union table's size 2 n0 + n_mg."""
    perm = np.random.default_rng(MG_SEED).permutation(n0)
    n_mg = int(perm.max()) // MG + 1
    return perm, n_mg, 2 * n0 + n_mg


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def union_table(E):
    """The union relation table (float32): the filed names as filed (rows 0..n0-1), mg3's merged names
    (chainscore19's Transform.names on the filed unit names: the unit sum of each group's), and the filed names again
    (rf's second halves keep r's name)."""
    if E.dtype not in (np.float16, np.float32):
        raise SystemExit(f"zt1gr: a relation table of {E.dtype}; float16 or float32 keeps the filed rows exact")
    n0 = E.shape[0]
    tr = CS19.Transform("mg3", n0, {"nodes": None}, None, None)
    if not np.array_equal(tr.perm, layout(n0)[0]):
        raise SystemExit("zt1gr: mg3's permutation is not chainscore19's")
    M = tr.names(unit(E.astype(np.float64)))
    return np.concatenate([E.astype(np.float32), np.asarray(M, np.float32), E.astype(np.float32)])


def h64(s):
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def regime(qid):
    return REGIMES[h64(f"t1:{qid}") % len(REGIMES)]


def reified(qid, n0):
    """(n0,) bool: the relations rf reifies in this question, each by its own hash (about half)."""
    return np.array([h64(f"t1rf:{qid}:{r}") % 2 == 0 for r in range(n0)])


def transform(g, qid, hl, tl, sl, n0):
    """One question's triples under regime g, in the union table's ids: (hl, tl, sl, M). rf's M mediators are
    numbered -1..-M in hl and tl (the pool's size is not known here; entries() places them after the pool), in
    chainscore19.reify_row's order."""
    if g == "id":
        return hl, tl, sl, 0
    perm, n_mg, _ = layout(n0)
    ok = sl >= 0
    if g == "mg3":
        out = np.full(sl.shape, -1, np.int64)
        out[ok] = n0 + perm[sl[ok].astype(np.int64)] // MG
        return hl, tl, out, 0
    if g != "rf":
        raise SystemExit(f"zt1gr: regime {g}")
    k = ok.sum(1)
    npk = int(max(hl.max(initial=-1), tl.max(initial=-1))) + 1
    key = np.unique((np.repeat(hl, k) * npk + np.repeat(tl, k)) * n0 + sl[ok].astype(np.int64))
    r = key % n0
    ht = key // n0
    h, t = ht // npk, ht % npk
    m = reified(qid, n0)[r]
    M = int(m.sum())
    med = -1 - np.arange(M, dtype=np.int64)
    return (np.r_[h[~m], h[m], med], np.r_[t[~m], med, t[m]],
            np.r_[r[~m], r[m], n0 + n_mg + r[m]][:, None], M)


# ── build (rmatch's and zrc's builds on the transformed triples) ─────────────


@contextlib.contextmanager
def on_build(ids, n0):
    """rmatch.build and zrc.build on T1's triples: row_triples gives each question's transformed triples (its regime
    by its id, in the carve's order), the entry functions walk the pool plus its mediators and keep the pool's rows
    only, and the look's relations are the union table's. Restored after."""
    _p, _m, n_u = layout(n0)
    st = {"i": 0, "pending": None, "regimes": {g: 0 for g in REGIMES}, "mediators": 0, "reified_triples": 0,
          "mediator_entries": 0}
    saved = RM.row_triples, RM.question_entries, ZC.contrib_entries, RM.look_relations

    def row_triples(c, ea, eb):
        if st["pending"] is not None:
            raise SystemExit("zt1gr: a question's triples were read twice before its entries")
        if st["i"] >= len(ids):
            raise SystemExit(f"zt1gr: more questions than the carve's {len(ids)} ids")
        qid = ids[st["i"]]
        g = regime(qid)
        hl, tl, sl = saved[0](c, ea, eb)
        if sl.size and int(sl.max()) >= n0:
            raise SystemExit(f"zt1gr: question {qid} has relation {int(sl.max())} of {n0}")
        hl, tl, sl, M = transform(g, qid, hl, tl, sl, n0)
        st["i"] += 1
        st["regimes"][g] += 1
        st["mediators"] += M
        st["reified_triples"] += M
        st["pending"] = M
        return hl, tl, sl

    def entries(fn):
        def wrapped(n, hl, tl, sl, seeds_by_bucket, kz, *a, **k):
            M = st["pending"]
            if M is None:
                raise SystemExit("zt1gr: entries before the question's triples")
            st["pending"] = None
            if kz != 2 * n_u:
                raise SystemExit(f"zt1gr: kz {kz}, the union table's is {2 * n_u}")
            hl = np.where(hl < 0, n - 1 - hl, hl)
            tl = np.where(tl < 0, n - 1 - tl, tl)
            real = np.r_[hl[hl < n], tl[tl < n]]
            if bool((real < 0).any()) or bool((np.r_[hl, tl] >= n + M).any()):
                raise SystemExit(f"zt1gr: a triple outside the pool of {n} and its {M} mediators")
            (row, zz, bb, mm), stats = fn(n + M, hl, tl, sl, seeds_by_bucket, kz, *a, **k)
            keep = row < n
            st["mediator_entries"] += int((~keep).sum())
            for b_, s_ in enumerate(stats):
                s_["entries"] = int((keep & (bb == b_)).sum())
            return (row[keep], zz[keep], bb[keep], mm[keep]), stats
        return wrapped

    def look_relations(d):
        rel = dict(saved[3](d))
        if int(rel["n_relations"]) != n0:
            raise SystemExit(f"zt1gr: the look has {rel['n_relations']} relations, the filed table {n0}")
        rel["n_relations"] = n_u
        return rel

    RM.row_triples, RM.question_entries = row_triples, entries(saved[1])
    ZC.contrib_entries, RM.look_relations = entries(saved[2]), look_relations
    try:
        yield st
        if st["i"] != len(ids) or st["pending"] is not None:
            raise SystemExit(f"zt1gr: the build read {st['i']} of {len(ids)} questions")
    finally:
        RM.row_triples, RM.question_entries, ZC.contrib_entries, RM.look_relations = saved


def check_order(ds, carve, cache_root, head, ids):
    """The cache's parts cover the look's chunks in order and its questions are the ids in order (the regimes are read
    by position)."""
    _parts, recs = LC.part_dirs(ds, carve, cache_root)
    at, qa = 0, 0
    for r in recs:
        lo, hi = r["chunks"]
        if lo != at or r["query_range"][0] != qa:
            raise SystemExit(f"zt1gr: {ds}/{carve}'s parts do not follow chunk {at}, question {qa}")
        at, qa = hi, r["query_range"][1]
    if at != head["n_chunks"] or qa != len(ids):
        raise SystemExit(f"zt1gr: the parts cover {at} of {head['n_chunks']} chunks, {qa} of {len(ids)} questions")


def write_table(path, T):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not np.array_equal(np.load(path), T):
            raise SystemExit(f"{path}: on disk and not this union table")
        return
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, T)
    os.replace(tmp, path)


def build(host=False, ds=DS, carve=CARVE, look_root=None, cache_root=None, rel_dir=None, out=OUT):
    t0 = time.time()
    look_root, cache_root = look_root or Z2.LOOK, cache_root or Z2.CACHE
    rel_dir, rt = Path(rel_dir or RM.REL_DIR), roots(out)
    d, _files, ids, head = LC.look_chunks(ds, carve, look_root)
    n0 = int(RM.look_relations(d)["n_relations"])
    ef = rel_dir / f"{ds}_rel_embeddings.npy"
    E = np.load(ef)
    if E.shape[0] != n0:
        raise SystemExit(f"{ef}: {E.shape[0]} relations, the look {n0}")
    T = union_table(E)
    write_table(rt["rel"] / ef.name, T)
    check_order(ds, carve, cache_root, head, ids)
    placement = {"where": "host" if host else "laptop", "t1": DECLARED}
    stats = {}
    with on_build(ids, n0) as st:
        RM.build(ds, carve, rt["rmatch"], cache_root=cache_root, look_root=look_root, rel_dir=rt["rel"],
                 placement=placement)
    stats["rmatch"] = st
    with on_build(ids, n0) as st:
        ZC.build(ds, carve, rt["zrc"], cache_root=cache_root, look_root=look_root, rel_dir=rt["rel"],
                 placement=placement, rm_root=rt["rmatch"])
    stats["zrc"] = st
    perm, n_mg, n_u = layout(n0)
    rec = {"stage": "T1", "declared_in": DECLARED, "dataset": ds, "carve": carve, "questions": len(ids),
           "ids_sha256": hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest(),
           "relations": {"filed": n0, "merged": n_mg, "union": n_u, "mg3_groups": [
               np.flatnonzero(perm // MG == j).tolist() for j in range(n_mg)],
               "filed_table_sha256": LC.sha_file(ef), "union_table_sha256": LC.sha_file(rt["rel"] / ef.name)},
           "builds": stats, "rmatch_records": RM.built_records(rt["rmatch"]),
           "zrc_records": RM.built_records(rt["zrc"]), "look_root": Path(look_root).as_posix(),
           "cache_root": Path(cache_root).as_posix(), "placement": placement,
           "zt1gr_sha256": LC.sha_src(__file__), "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if stats["rmatch"]["regimes"] != stats["zrc"]["regimes"] or stats["rmatch"]["mediators"] != stats["zrc"]["mediators"]:
        raise SystemExit("zt1gr: the two builds disagree on the regimes")
    LC.write_json(rt["build"], rec)
    log(f"zt1gr build {ds}/{carve}: regimes {stats['rmatch']['regimes']}, {stats['rmatch']['mediators']} mediators "
        f"({time.time() - t0:.0f}s)")
    return rec


# ── train: metaqa's fit carve reads T1's chains ──────────────────────────────


@contextlib.contextmanager
def on_t1(swap=None, carve=(DS, CARVE), rel_t1=None):
    """While open, the carve (metaqa's fit carve) loads its chains from T1's roots (rmatch's root -> T1's rmatch
    chains, zrc's -> T1's zrc chains) and T1's union table, its centred table centred on the filed relations' mean;
    every other carve loads as filed. Restored after."""
    swap = {Path(k).resolve(): Path(v) for k, v in (swap or {Z2.CH_RM: roots()["rmatch"],
                                                            Z2.CH_ZRC: roots()["zrc"]}).items()}
    rel_t1 = Path(rel_t1 or roots()["rel"])
    load = RM.load_chains

    def t1_load(c, ch_root=RM.CH_OUT, rel_dir=RM.REL_DIR, cache_root=LC.OUT, verify=True):
        if (c.ds, c.carve) != tuple(carve):
            return load(c, ch_root, rel_dir, cache_root, verify)
        to = swap.get(Path(ch_root).resolve())
        if to is None:
            raise SystemExit(f"zt1gr: {c.ds}/{c.carve} from {ch_root}, which T1 does not replace")
        E0 = np.load(Path(rel_dir) / f"{c.ds}_rel_embeddings.npy")
        T = union_table(E0)
        tf = rel_t1 / f"{c.ds}_rel_embeddings.npy"
        if not np.array_equal(np.load(tf), T):
            raise SystemExit(f"{tf}: not the union of {Path(rel_dir) / tf.name}")
        ch = load(c, to, rel_t1, cache_root, verify)
        if ch["n_rel"] != T.shape[0]:
            raise SystemExit(f"zt1gr: {to} was built on {ch['n_rel']} relations, the union table has {T.shape[0]}")
        U = unit(T.astype(np.float64))
        ch["rel_centered"] = torch.from_numpy((U - U[:E0.shape[0]].mean(0, keepdims=True)).astype(np.float32)).to(
            c.device)
        LOADED.append({"dataset": c.ds, "carve": c.carve, "from": Path(ch_root).as_posix(), "chains": to.as_posix(),
                       "records": ch["records"], "union_table_sha256": LC.sha_file(tf)})
        return ch

    RM.load_chains = t1_load
    try:
        yield LOADED
    finally:
        RM.load_chains = load


def stamp(argv, arm, split, build_rec):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zt1gr_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "t1": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "device": Z2.DEVICE,
                        "augmentation": f"{DS}/{CARVE} in training: one regime per question ({', '.join(REGIMES)})",
                        "regimes": build_rec["builds"]["rmatch"]["regimes"],
                        "build_sha256": LC.sha_file(roots()["build"]), "loaded": list(LOADED),
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def built():
    bp = roots()["build"]
    if not bp.exists():
        raise SystemExit(f"zt1gr: {bp} missing (zt1gr.py build)")
    rec = json.loads(bp.read_text(encoding="utf-8"))
    if rec.get("stage") != "T1" or (rec.get("dataset"), rec.get("carve")) != (DS, CARVE):
        raise SystemExit(f"zt1gr: {bp} is not T1's build of {DS}/{CARVE}")
    return rec


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zt1gr: train takes --arm zrckpt or zspkt, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zt1gr: T1's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zt1gr: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    brec = built()
    Z3.kt_records(Z3.train_carves(split))
    LOADED.clear()
    with Z2.on_p1c2(), on_t1():
        rc = RM.train(argv, split)
    if not any((x["dataset"], x["carve"]) == (DS, CARVE) for x in LOADED):
        raise SystemExit(f"zt1gr: {DS}/{CARVE} was not read under T1's chains")
    with Z2.on_p1c2():
        stamp(argv, arm, split, brec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") not in ARMS or "t1" not in rec:
        raise SystemExit(f"zt1gr: {name} is not a T1 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fit ────────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zt1gr: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp, metaqa as filed)"),
            ("(rel's screen fits)", f"({b}'s card fits, metaqa as filed)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (T1's metaqa) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"T1 ({DECLARED}): {arm} ({BASE_OF[arm]} trained with metaqa's granularity transforms) against "
                      f"{BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "T1", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "zt1gr_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def verdict(recalls=None, out=None):
    """T1's screen verdict, per arm, from its re-call (section 3); webqsp (the target) and metaqa beside."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems = Path(recalls[a]), []
        rows = []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "T1" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not T1's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in ("webqsp", DS)},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    rec = {"stage": "T1", "declared_in": DECLARED, "rule": RULE, "arms": per, "zt1gr_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# T1's screen verdict: zrckpt **{per['zrckpt']['verdict']}**, zspkt **{per['zspkt']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {v['call']} ({v['delta']:+.4f})" if isinstance(v["delta"], (int, float))
                               else f"{k} {v['call']}" for k, v in p["beside"].items()) or "none") + ".")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zt1gr verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "T1", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def toy_seeds(cq):
    sl0, bk = cq["q_seed_local"], cq["q_seed_bucket"]
    return [np.unique(sl0[(sl0 >= 0) & (bk == b)]).astype(np.int64) for b in range(RM.BUCKETS)]


def by_hand(fn, qid, cq, n0, *extra):
    """One toy question's entries under its regime, from chainscore19's own transforms: mg3's relabel, and rf's
    reify_row with this question's reified half (its second halves moved past the merged names)."""
    n = int(cq["q_pool_size"])
    hl, tl, sl = RM.row_triples(cq, 0, int(cq["q_edges"]))
    g = regime(qid)
    perm, n_mg, n_u = layout(n0)
    if g == "mg3":
        tr = CS19.Transform("mg3", n0, {"nodes": None}, None, None)
        assert np.array_equal(tr.perm, perm)
        s2 = tr.relabel(sl, None, None)
        sl = np.where(s2 >= 0, s2 + n0, -1)
    elif g == "rf":
        tr = CS19.Transform("rf", n0, {"nodes": None}, None, np.ones(n0, bool))
        tr.reif = reified(qid, n0)
        n, hl, tl, s2, _pn, _rr = tr.reify_row(n, hl, tl, sl, np.ones((n, 2), np.float32), np.zeros(n))
        sl = np.where(s2 >= n0, s2 + n_mg, s2)
    (row, zz, bb, mm), _st = fn(n, hl, tl, sl, toy_seeds(cq), 2 * n_u, *extra)
    keep = row < int(cq["q_pool_size"])
    return row[keep], zz[keep], bb[keep], mm[keep], np.unique(sl[sl >= 0])


def selftest():
    import tempfile
    t0 = time.time()
    # 1. the arms are the lean pair's, unchanged
    assert S.ARMS["zrckpt"] == S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC)
    assert S.ARMS["zspkt"] == S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    # 2. the union table: the filed rows exact, mg3's names chainscore19's, rf's second halves the filed names
    rng = np.random.default_rng(11)
    E = rng.standard_normal((9, RM.Q_DIM)).astype(np.float16)
    T = union_table(E)
    perm, n_mg, n_u = layout(9)
    assert (n_mg, n_u, T.shape) == (3, 21, (21, RM.Q_DIM)) and np.array_equal(T[:9], E.astype(np.float32))
    assert np.array_equal(T[12:], E.astype(np.float32))
    tr = CS19.Transform("mg3", 9, {"nodes": None}, None, None)
    assert np.array_equal(T[9:12], tr.names(unit(E.astype(np.float64))).astype(np.float32))
    assert sorted(np.bincount(perm // MG).tolist()) == [3, 3, 3]
    cnt = {g: sum(regime(f"x{i}") == g for i in range(3000)) for g in REGIMES}
    assert all(900 < v < 1100 for v in cnt.values()), cnt
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        # 3. a toy build: every question's entries are rmatch's and zrc's on chainscore19's transform of its triples,
        #    the pool's rows only; an id question's are the filed build's, array for array
        qs_arr = RM.toy_roots(T_, rng, n_rel=5, chunks=(6, 6))
        ZC.add_q_emb(T_, rng)
        base = {"cache_root": T_ / "cache", "look_root": T_ / "look", "rel_dir": T_ / "rel"}
        RM.build("metaqa", "toy", T_ / "ch", **base)
        ZC.build("metaqa", "toy", T_ / "zc", **base, rm_root=T_ / "ch")
        ids = [f"q{i}" for i in range(len(qs_arr))]
        regs = [regime(q) for q in ids]
        assert set(regs) == set(REGIMES), regs
        rec = build(ds="metaqa", carve="toy", out=T_ / "t1", **base)
        rt = roots(T_ / "t1")
        assert rec["builds"]["rmatch"]["regimes"] == {g: regs.count(g) for g in REGIMES}
        assert rec["builds"]["rmatch"]["mediators"] > 0 and rec["relations"]["union"] == 2 * 5 + 2
        U5 = unit(np.load(T_ / "rel" / "metaqa_rel_embeddings.npy").astype(np.float64))
        TU = unit(np.load(rt["rel"] / "metaqa_rel_embeddings.npy").astype(np.float64))
        assert np.array_equal(TU[:5], U5)
        with np.load(sorted((T_ / "look" / "metaqa" / "toy" / "chunks").glob("c*.npz"))[0]) as zf:
            qe0 = zf["q_emb"]
        with np.load(sorted((T_ / "look" / "metaqa" / "toy" / "chunks").glob("c*.npz"))[1]) as zf:
            qe1 = zf["q_emb"]
        q_emb = np.concatenate([qe0, qe1])

        def stored(base_root, t1):
            Cls = RM.chain_carve(RM.ToyBase, base_root, T_ / "rel")
            if not t1:
                return Cls("metaqa", "toy", "2wiki", "cpu", T_ / "cache").chains
            with on_t1({T_ / "ch": rt["rmatch"], T_ / "zc": rt["zrc"]}, ("metaqa", "toy"), rt["rel"]):
                return Cls("metaqa", "toy", "2wiki", "cpu", T_ / "cache").chains

        ch_rm, ch_zc = stored(T_ / "ch", True), stored(T_ / "zc", True)
        f_rm, f_zc = stored(T_ / "ch", False), stored(T_ / "zc", False)
        same_id = 0
        for qi, (qid, tq) in enumerate(zip(ids, qs_arr)):
            cq = {k: np.asarray(v) for k, v in tq.items()}
            want = by_hand(RM.question_entries, qid, cq, 5)
            hl, tl, sl = RM.row_triples(cq, 0, int(cq["q_edges"]))
            qr = np.unique(by_hand(RM.question_entries, qid, cq, 5)[4])
            step = ZC.step_of(q_emb[qi], TU, qr)
            want_z = by_hand(ZC.contrib_entries, qid, cq, 5, step)
            for ch, w in ((ch_rm, want), (ch_zc, want_z)):
                e0, e1 = ch["e_off"][qi], ch["e_off"][qi + 1]
                assert np.array_equal(ch["ent_row"][e0:e1], w[0]) and np.array_equal(ch["ent_z"][e0:e1], w[1])
                assert np.array_equal(ch["ent_b"][e0:e1], w[2])
                assert np.array_equal(ch["ent_m"][e0:e1], w[3].astype(np.float16))
                assert np.array_equal(ch["q_rel"][ch["r_off"][qi]:ch["r_off"][qi + 1]], w[4])
            if regs[qi] == "id":
                for ch, f in ((ch_rm, f_rm), (ch_zc, f_zc)):
                    for k in ("ent_row", "ent_z", "ent_b", "ent_m"):
                        assert np.array_equal(ch[k][ch["e_off"][qi]:ch["e_off"][qi + 1]],
                                              f[k][f["e_off"][qi]:f["e_off"][qi + 1]]), k
                same_id += 1
        assert same_id == regs.count("id")
        # 4. the load: rel_unit's and rel_centered's filed rows are the filed tables', so an id question's chain
        #    features are the filed carve's; another carve, and the loader after, are as filed
        assert torch.equal(ch_rm["rel_unit"][:5], f_rm["rel_unit"]) and torch.equal(ch_rm["rel_centered"][:5],
                                                                                    f_rm["rel_centered"])
        assert ch_rm["n_rel"] == 12 and len(LOADED) >= 2 and LOADED[-1]["chains"] == rt["zrc"].as_posix()
        load = RM.load_chains
        with on_t1({T_ / "ch": rt["rmatch"]}, ("metaqa", "other"), rt["rel"]):
            c = RM.chain_carve(RM.ToyBase, T_ / "ch", T_ / "rel")("metaqa", "toy", "2wiki", "cpu", T_ / "cache")
            assert c.chains["n_rel"] == 5 and RM.load_chains is not load
        assert RM.load_chains is load

        def bad_root():
            with on_t1({T_ / "zc": rt["zrc"]}, ("metaqa", "toy"), rt["rel"]):
                RM.chain_carve(RM.ToyBase, T_ / "ch", T_ / "rel")("metaqa", "toy", "2wiki", "cpu", T_ / "cache")

        SR.must_stop(bad_root)
        assert RM.load_chains is load
        idq = np.array([i for i, g in enumerate(regs) if g == "id"])
        allq = np.arange(len(ids))
        Cb = RM.chain_carve(RM.ToyBase, T_ / "ch", T_ / "rel")("metaqa", "toy", "2wiki", "cpu", T_ / "cache")
        with on_t1({T_ / "ch": rt["rmatch"]}, ("metaqa", "toy"), rt["rel"]):
            Ct = RM.chain_carve(RM.ToyBase, T_ / "ch", T_ / "rel")("metaqa", "toy", "2wiki", "cpu", T_ / "cache")
        torch.manual_seed(0)
        cm = RM.ChainMatch(["rank", "SEMB"], Cb.widths, 16, dropout=0.1, seed=0).eval()
        with torch.no_grad():
            gen = torch.Generator().manual_seed(3)
            for k, p in cm.named_parameters():
                if k.startswith("cm_"):
                    p.copy_(torch.randn(p.shape, generator=gen) * (0.02 if k in ("cm_b", "cm_pi_w") else 0.5))
            fb, _nq, _bz, _g = Cb.batch(idq, ["rank", "SEMB"])
            ft, _nq2, _bz2, _g2 = Ct.batch(idq, ["rank", "SEMB"])
            nrow = int(Cb.n_np[idq].sum())
            a_, b_ = cm.chain_feats(fb, nrow), cm.chain_feats(ft, nrow)
            gap = max(float((a_[j] - b_[j]).abs().max()) for j in (0, 1))
            assert gap <= 1e-6 * max(1.0, float(a_[0].abs().max())), gap
            log(f"zt1gr selftest: an id question's chain features against the filed carve's: max gap {gap:.2e}")
            fa, _n, _b, _gg = Ct.batch(allq, ["rank", "SEMB"])
            fz, _n, _b, _gg = Cb.batch(allq, ["rank", "SEMB"])
            n_all = int(Cb.n_np.sum())
            assert not torch.equal(cm.chain_feats(fa, n_all)[0], cm.chain_feats(fz, n_all)[0])
        # 5. the refusals
        good = ["train", "--arm", "zrckpt", "--name", "scr-t1-zrckpt", "--cache-root", "outputs/p1c2/cache"]
        SR.must_stop(train, ["train", "--arm", "zrckp", "--name", "scr-t1-zrckpt", "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, good, "L-2wiki")
        SR.must_stop(train, ["train", "--arm", "zrckpt", "--name", "scr-t1-zrckpt-hp", "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, ["train", "--arm", "zspkt", "--name", "scr-t1-zspkt", "--cache-root",
                             "outputs/p1gb/cache"], "L-musique")
        SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
        (T_ / "fits" / "y").mkdir(parents=True)
        LC.write_json(T_ / "fits" / "y" / "screen.json", {"arm": "zrckp", "p1c3p": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        SR.must_stop(write_table, rt["rel"] / "metaqa_rel_embeddings.npy", T[:12])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T_ / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T_ / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147},
                                 arm=arm, base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T_ / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T_ / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "T1" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"T1 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (T1's metaqa) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm
        gm, gw = ("L-musique", "musique"), ("L-hotpotqa", "webqsp")
        rec = verdict({"zrckpt": fake_recall(T_ / "a.json", "zrckpt", {gw: "GAIN"}),
                       "zspkt": fake_recall(T_ / "b.json", "zspkt", {gm: "GAIN", gw: "LOSS"})}, T_ / "v")
        assert rec["arms"]["zrckpt"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspkt"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zrckpt"]["beside"]["L-hotpotqa webqsp"]["call"] == "GAIN"
        rec = verdict({"zrckpt": fake_recall(T_ / "a.json", "zrckpt", {}),
                       "zspkt": fake_recall(T_ / "b.json", "zspkt", {gm: "GAIN"}, stage="L1")}, T_ / "v")
        assert rec["arms"]["zrckpt"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkt"]["verdict"] == "INCOMPLETE"
        rec = verdict({"zrckpt": T_ / "none.json", "zspkt": fake_recall(T_ / "b.json", "zrckpt", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpt"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkt"]["verdict"] == "INCOMPLETE"
    log(f"zt1gr selftest: the arms are the lean pair's; the union table keeps the filed rows exact, mg3's names are "
        f"chainscore19's; on a toy carve every question's rmatch and zrc entries are those of chainscore19's mg3 and "
        f"rf on its triples (pool rows only) and an id question's are the filed build's; the filed rows of both tables "
        f"load exact, so an id question's chain features are the filed carve's to float32 rounding; only the named carve is swapped and "
        f"the loader is restored; train refuses another arm, split, fit name, cache root or device, read a fit that "
        f"is not T1's; relz's pair and re-call stamped T1's; the verdict per arm ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "build":
        build(host="--host" in rest)
        return 0
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"zt1gr {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zt1gr: build, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
