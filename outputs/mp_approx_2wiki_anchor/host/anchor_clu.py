"""Design look (untracked; not a result and not filed): cluster-typed relations, so a relation never seen in training
(a held phrase, another dataset's anchor phrase, a KB relation name) still gets a trained type through its text.

The source dataset's top N phrases (anchor_gen3's 4096-row table) are grouped into M text clusters by spherical k-means
(l16_look_analyze.kmeans_sph, seed 0, 30 iterations). With a hold, the held fold's phrases are left out of the clustering,
so a held phrase's cluster is chosen by its text alone, as an unseen phrase's would be. Phrase rank r < N maps to its
nearest centroid c(r) = argmax_c <phi(r), C_c>; ranks >= N, phrases with no text, and 'other' map to K. A model is any
anchor_gen3 model with K = 256 or 1024 token slots; the first M are the clusters, and the rest never occur. Text and gen
models read the centroids as their phrase table. Training and every read use one rule, so any graph whose edge labels
can be embedded reads with no string match.

    fit    anchor_gen3's fit with anchor_gen's identity_map / hold_map / held_reach_rows and anchor_gen3's phrase_table /
           save_model rebound in-process to the cluster map, the centroid table and a checkpoint carrying the clusters.
           anchor_gen3 reads ID (MASK with a hold: held phrases -> other) and NR. Then this file's own read follows in-domain.
    read   any target among 2wiki, hotpotqa, musique, metaqa and webqsp (anchor_nn's loader):
           BASE   every phrase / relation name -> its nearest centroid (in-domain: the training map, checked)
           NR     every edge 'other'
           MASK   in-domain with a hold: held phrases -> other (the fit's own read)
           REV    in-domain with a hold: held phrases -> their nearest centroid (= BASE), on all rows and on the held-reach rows
           SHUF   on a KB: each relation given the next relation's cluster (a cyclic derangement)
--mres doubles every view (anchor_mres), in training and in reading; a model fitted with it is read with it.

    python outputs/mp_approx_2wiki_anchor/host/anchor_clu.py fit --dataset 2wiki --variants x4+x5+x6:A256-1 --m 64 --out PATH [--hold 0/5] [--mres]
    python outputs/mp_approx_2wiki_anchor/host/anchor_clu.py read --target metaqa --models a.pt,b.pt --out PATH [--mres]
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
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
import anchor_nn as NN  # noqa: E402

AG = G3.AG
AW, AW3, AW6, A16 = AG.AW, AG.AW3, AG.AW6, AG.A16
R_TOP = AG.R_TOP
PINS = {"anchor_gen3": "ee5f76dc7d391649e04ab2b3575d4fdd30dd62cd2a34b1b7dbd114d56f7b14a7",
        "anchor_gen_kb": "4e09d59de0a60767eb591863352389ff99e164f0e2cd841b858debd309ee7952",
        "anchor_nn": "e3bf4e1f1598ac4d62eadc58cefbcc1056d27e8d812c7e11c33c8b31ccb8678c",
        "anchor_mres": "e9dcc06e043bd590c14b5586adcdefe94eb8745a1444f1f3d52d122fbc8b93fd"}
SEED, ITERS = 0, 30
MS = (16, 32, 64, 128, 256, 512, 1024)
log = AG.log
_orig = {"identity_map": AG.identity_map, "hold_map": AG.hold_map, "held_reach_rows": AG.held_reach_rows,
         "phrase_table": G3.phrase_table, "save_model": G3.save_model}


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def assign(phi, C, n):
    """Each of the table's first n rows -> its nearest centroid, R_TOP + 1 entries; a row with no text and every rank at
    or past n -> -1."""
    lab = np.full(R_TOP + 1, -1, dtype=np.int64)
    n = min(n, phi.shape[0], R_TOP)
    X = np.asarray(phi[:n], dtype=np.float32)
    ok = np.linalg.norm(X, axis=1) > 1e-6
    if ok.any():
        lab[np.flatnonzero(ok)] = np.argmax(unit_rows(X[ok]) @ C.T, 1)
    return lab


def cluster_map(lab, K, held=None):
    rm = np.where(lab >= 0, lab, K).astype(np.int64)
    rm[R_TOP] = K
    if held is not None:
        rm[held] = K
    return rm


def held_fold(vocab, n, h, F):
    held = np.zeros(R_TOP + 1, dtype=bool)
    for r in range(min(n, len(vocab), R_TOP)):
        if AG.fold_of(vocab[r], F) == h:
            held[r] = True
    return held


def source_vocab():
    """The rebound dataset's phrase strings by rank: anchor_walk.anchor_tables' w2 list, read before any row is loaded."""
    if AW.COMPACT_SHA is None or AW.sha(AW.COMPACT) != AW.COMPACT_SHA:
        raise SystemExit("anchors_compact.npz is not the extracted one")
    with np.load(AW.COMPACT) as zf:
        return [str(s) for s in zf["w2_top"]]


def build_clusters(phi, vocab, M, N, held):
    """Centroids from the top N phrases with text (held ones left out), and each cluster's five commonest members."""
    X = np.asarray(phi[:N], dtype=np.float32)
    keep = np.linalg.norm(X, axis=1) > 1e-6
    if held is not None:
        keep &= ~held[:N]
    if keep.sum() < M:
        raise SystemExit(f"{int(keep.sum())} phrases for {M} clusters")
    C = A16.kmeans_sph(X[keep], M, seed=SEED, iters=ITERS).astype(np.float32)
    lab = assign(phi, C, N)
    sizes = np.bincount(lab[lab >= 0], minlength=M)
    members = {c: [vocab[r] for r in np.flatnonzero(lab[:N] == c)[:5]] for c in range(M)}
    return C, lab, sizes, members


def centroid_table(C):
    T = np.zeros((R_TOP, C.shape[1]), dtype=np.float32)
    T[:C.shape[0]] = C
    return T


def mres_install(on):
    if on:
        import anchor_mres as MR
        if AW.sha(Path(MR.__file__)) != PINS["anchor_mres"]:
            raise SystemExit("anchor_mres.py is not the pinned file")
        MR.install()
        return MR
    return None


def check_pins():
    for mod, key in ((G3, "anchor_gen3"), (KB, "anchor_gen_kb"), (NN, "anchor_nn")):
        if AW.sha(Path(mod.__file__)) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")


def main_fit(a, rest):
    ds = a.dataset
    if a.m not in MS:
        raise SystemExit(f"--m among {MS}")
    hold = tuple(int(x) for x in a.hold.split("/")) if a.hold else None
    if a.rdrop_row or a.rdrop_edge:
        raise SystemExit("relation dropout is not read on clusters (its K counts ranks, not clusters)")
    train_looks = AW6.TRAIN[ds] if ds != "2wiki" else ("fit", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12")
    for name in a.variants.split(","):
        sp = G3.parse(name, train_looks)
        if sp["pca"] or sp["K"] < a.m:
            raise SystemExit(f"{name}: no PCA code, and K >= M")
    AW6.rebind(ds)
    vocab = source_vocab()
    phi, phi_sha = _orig["phrase_table"](ds)
    N = min(a.n, R_TOP)
    held = held_fold(vocab, N, *hold) if hold else None
    C, lab, sizes, members = build_clusters(phi, vocab, a.m, N, held)
    T = centroid_table(C)
    t_sha = hashlib.sha256(T.tobytes()).hexdigest()
    clu = {"M": a.m, "N": N, "seed": SEED, "iters": ITERS, "source": ds, "source_table": phi_sha, "held_excluded": held is not None,
           "hold": a.hold, "mres": bool(a.mres), "centroid_table_sha256": t_sha, "C": C, "lab": lab}
    log(f"{a.m} clusters over {N} phrases ({int((lab >= 0).sum())} with text, {0 if held is None else int(held.sum())} held): "
        f"sizes min {int(sizes.min())} median {int(np.median(sizes))} max {int(sizes.max())}")

    def identity_map(K):
        if K < a.m:
            raise SystemExit(f"K {K} < M {a.m}")
        return cluster_map(lab, K)

    def hold_map(K, vocab_w2, h, F):
        hd = held_fold(vocab_w2, N, h, F)
        if held is None or not np.array_equal(hd, held):
            raise SystemExit("the fit's hold is not the clustering's")
        return cluster_map(lab, K, hd), hd

    def held_reach_rows(Q, rows, hd, K):
        return _orig["held_reach_rows"](Q, rows, hd, max(K, N))

    def phrase_table(ds_):
        if ds_ != ds:
            raise SystemExit("the centroid table is the source dataset's")
        return T, t_sha

    def save_model(path, model, sp, nt, margin, ds_, vocab_w2, phi_sha_, extra):
        if sp.get("pca"):
            raise SystemExit("the PCA code is not read on clusters")
        _orig["save_model"](path, model, sp, nt, margin, ds_, vocab_w2, phi_sha_, {**extra, "clusters": clu})

    AG.identity_map, AG.hold_map, AG.held_reach_rows = identity_map, hold_map, held_reach_rows
    G3.phrase_table, G3.save_model = phrase_table, save_model
    G3.main(rest)
    AG.identity_map, AG.hold_map, AG.held_reach_rows = _orig["identity_map"], _orig["hold_map"], _orig["held_reach_rows"]
    G3.phrase_table, G3.save_model = _orig["phrase_table"], _orig["save_model"]
    out = Path(a.out)
    r = json.loads(out.read_text(encoding="utf-8"))
    r["clu"] = {k: v for k, v in clu.items() if k not in ("C", "lab")} | {"sizes": sizes.tolist(), "members": members, "script_sha256": AW.sha(Path(__file__)),
                                                           "pins": PINS, "mres_doubled": dict(a.MR.STATS) if a.MR else None}
    out.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    pts = sorted((out.parent / (out.stem + "_models")).glob("v*_s*.pt"))
    run_read(Namespace(target=ds, models=",".join(str(p) for p in pts), out=str(out.with_name(out.stem + "_read.json")), mres=a.mres, MR=a.MR))


class Namespace(argparse.Namespace):
    pass


def load_clu_model(p, ck):
    sp, cl = ck["spec"], ck["clusters"]
    T = centroid_table(np.asarray(cl["C"], dtype=np.float32))
    if hashlib.sha256(T.tobytes()).hexdigest() != cl["centroid_table_sha256"]:
        raise SystemExit(f"{p}: the stored centroids do not match their sha256")
    phi = T if sp["family"] != "free" else None
    model, _ck = (G3.load_model if "pca" in sp else AG.load_model)(p, phi)
    return model, T


def run_read(a):
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    paths = [Path(p) for p in a.models.split(",")]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for p, ck in zip(paths, cks):
        if "clusters" not in ck:
            raise SystemExit(f"{p} was not fitted here")
        if bool(ck["clusters"]["mres"]) != bool(a.mres):
            raise SystemExit(f"{p}: fitted with mres={ck['clusters']['mres']}, read with mres={bool(a.mres)}")
    max_len = max(ck["spec"]["max_len"] for ck in cks)
    Q, part, vocab_t, phi_t, _own, info = NN.load_target(a.target, max_len, False, t0)
    kb = info["kind"] == "kb"
    x1 = part["x1"]
    B_rows = x1[1::2]
    RB = AG.Reader(Q, B_rows, 20261002)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "anchor_clu", "mode": "read", "target": a.target, "script_sha256": AW.sha(Path(__file__)), "pins": PINS, "mres": bool(a.mres),
           "target_info": info, "B": RB.base(), "models": {}}
    per_row = {}
    pos = {i: j for j, i in enumerate(B_rows)}
    for jm, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp, K, cl = ck["spec"], ck["spec"]["K"], ck["clusters"]
        C = np.asarray(cl["C"], dtype=np.float32)
        model, _T = load_clu_model(p, ck)
        in_domain = ck["dataset"] == a.target
        lab_t = assign(phi_t, C, cl["N"])
        ent = {"from": ck["dataset"], "name": ck.get("name"), "seed": ck.get("seed"), "hold": ck.get("hold"), "rdrop": ck.get("rdrop"),
               "spec": sp, "M": cl["M"], "N": cl["N"], "margin": ck["margin"], "model_file": str(p), "in_domain": in_domain, "reads": {}}
        if in_domain and not np.array_equal(lab_t, cl["lab"]):
            raise SystemExit(f"{p}: the target's phrases do not reproduce the training clusters")
        held = held_fold(vocab_t, cl["N"], *(int(x) for x in ck["hold"].split("/"))) if (in_domain and ck.get("hold")) else None
        maps = {"BASE": cluster_map(lab_t, K), "NR": np.full(R_TOP + 1, K, dtype=np.int64)}
        if held is not None:
            maps["MASK"] = cluster_map(lab_t, K, held)
        if kb:
            n = min(len(vocab_t), cl["N"], R_TOP)
            sh = maps["BASE"].copy()
            sh[:n] = maps["BASE"][(np.arange(n) + 1) % n]
            maps["SHUF"] = sh
            ent["kb_clusters"] = {vocab_t[r]: int(lab_t[r]) for r in range(n)}
        ent["typed_target_ranks"] = int((lab_t[:R_TOP] >= 0).sum())
        reads = {}
        for nm, rm in maps.items():
            _nt, TY = AG.types_for(Q, sp, rm, B_rows)
            reads[nm] = AG.read(model, Q, TY, ck["nt"], B_rows, z_of, ck["margin"])
            per_row[f"{jm}_B_{nm}"] = reads[nm]
        main = "MASK" if "MASK" in reads else "BASE"
        hrows = AG.held_reach_rows(Q, B_rows, held, cl["N"]) if held is not None else []
        hx = np.asarray([pos[i] for i in hrows], dtype=np.int64)
        for nm, m in reads.items():
            e = RB.record(m, by_type=False)
            e["typed_edge_share_B"] = NN.typed_share(Q, B_rows, maps[nm], K)
            if nm != main:
                e[f"minus_{main}"] = AW3.boot_pair(m - reads[main], RB.W)
            if hx.size:
                e["held_reach"] = RB.subset(m, hrows)
                if nm != main:
                    e[f"held_reach_minus_{main}"] = AW3.boot_pair((m - reads[main])[hx], RB.W[:, hx])
            ent["reads"][nm] = e
        if held is not None:
            ent["held"] = {"phrases": int(held.sum()), "held_reach_rows": len(hrows)}
        ent["seconds"] = round(time.time() - t1, 1)
        res["models"][f"{p.parent.name}/{p.name}"] = ent
        log(f"{p.name} ({ck['dataset']} -> {a.target}, {sp['model']}, M {cl['M']}): " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in ent["reads"].items()))
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))
    if a.MR:
        res["mres_doubled"] = dict(a.MR.STATS)
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("fit", "read"):
        raise SystemExit("mode first: fit or read")
    mode, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--mres", action="store_true")
    if mode == "fit":
        ap.add_argument("--dataset", default="2wiki")
        ap.add_argument("--m", type=int, required=True)
        ap.add_argument("--n", type=int, default=R_TOP)
        ap.add_argument("--hold", default=None)
        ap.add_argument("--out", required=True)
        ap.add_argument("--variants", required=True)
        ap.add_argument("--rdrop_row", type=float, default=0.0)
        ap.add_argument("--rdrop_edge", type=float, default=0.0)
        a, _unk = ap.parse_known_args(rest)
        rest = [x for x in rest if x != "--mres"]
        for flag in ("--m", "--n"):
            if flag in rest:
                i = rest.index(flag)
                rest = rest[:i] + rest[i + 2:]
    else:
        ap.add_argument("--target", required=True)
        ap.add_argument("--models", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args(rest)
    if "clu" not in Path(a.out).name or (a.mres and "mres" not in Path(a.out).name):
        raise SystemExit("--out must carry 'clu' (and 'mres' with --mres) in its file name")
    check_pins()
    a.MR = mres_install(a.mres)
    if mode == "fit":
        main_fit(a, rest)
    else:
        run_read(a)


if __name__ == "__main__":
    main()
