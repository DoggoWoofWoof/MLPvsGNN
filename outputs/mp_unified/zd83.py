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

The steps that read the links' labels or the recomputed KT rows (links, typed, kt, record) and the chain that runs them are
refused by this version: their code is committed, self-tested, before they run (docs section 10).
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
DEFERRED = ("links", "typed", "kt", "record")                       # their D83 code is committed before they run
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
        if old.get("sha256") != ent["sha256"] or old.get("kt", {}).get("sha256") != ent["kt"]["sha256"]:
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
    if a.cmd in ("chain", "step", "link", "build"):
        if not a.dataset:
            raise SystemExit(f"{a.cmd}: --dataset is required")
        use(a.arm, a.dataset)
    if a.cmd == "link":
        return link_pools(a.arm, a.dataset, a.host)
    if a.cmd == "build":
        return build(a.arm, a.dataset, a.lanes, a.host)
    if a.cmd == "chain":
        raise SystemExit("D83: the chain's later steps (links with labels, the KT pass, typed, kt, record) wait on their code")
    if a.cmd == "step":
        if a.kind in ("pools", "merge"):
            raise SystemExit("D83: the pools are the record's (A2); pools and merge never run")
        if a.kind in DEFERRED:
            raise SystemExit(f"D83: step {a.kind} waits on its D83 code (labels, the KT pass); nothing run")
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
                   lambda: Z.pools("D83F", "musique"), lambda: main(["step", "pools", "--arm", "D83F", "--dataset", "squad"]),
                   lambda: main(["chain", "--arm", "D83F", "--dataset", "squad"])]
        refused += [lambda k=k: main(["step", k, "--arm", "D83F", "--dataset", "squad"]) for k in DEFERRED]
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
    print("zd83 self-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
