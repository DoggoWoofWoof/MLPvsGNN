"""Design look (untracked; not a result and not filed): one cluster-typed relation model fitted on several graphs at
once, so its relation types are learned from more than one graph's phrases, then read zero-shot on graphs it never saw.

The training datasets' top N phrase texts are pooled (with a hold, the held fold of every dataset left out, by string) and
grouped into M clusters by spherical k-means (anchor_clu's seed and iterations). Every dataset's phrase rank maps to its
nearest centroid (anchor_clu's rule), so a type means one region of text on every graph. The fit is anchor_walk8's loop on
the union of the datasets' training looks, its epoch chosen on the union of their select carves. Each training dataset is
read on its own x1 B: ID (MASK with a hold: held phrases -> other), NR (every edge other), and with a hold REV (held
phrases -> their nearest centroid), on all rows and on the held-reach rows. The checkpoint carries anchor_clu's
'clusters' block, so anchor_clu.py's read mode reads it on any other dataset or KB (BASE, NR, SHUF).
Datasets load in the order 2wiki, hotpotqa, musique: anchor_walk6.rebind('2wiki') changes nothing, so 2wiki loads first.
--mres doubles every view (anchor_mres) in training and reading.

    python outputs/mp_approx_2wiki_anchor/host/anchor_joint.py --datasets 2wiki,hotpotqa --train 2wiki=x4+x5+x6,hotpotqa=x4+x5+x6 \
        --variants A256-1 --m 64 --out PATH [--hold 0/5] [--mres] [--lr 1e-3 --kd 1 --ce 1]
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
import anchor_clu as CL  # noqa: E402

G3, AG = CL.G3, CL.AG
AW, AW3, AW6, AW8, A16 = AG.AW, AG.AW3, AG.AW6, AG.AW8, AG.A16
R_TOP = AG.R_TOP
PINS = {"anchor_clu": "23479cd7100fac8223a281f132d62cefa4fe734637f09408195c2d0ddef62d1a", **CL.PINS}
ORDER = ("2wiki", "hotpotqa", "musique")
log = AG.log


def train_looks(ds):
    return AW6.TRAIN[ds] if ds != "2wiki" else ("fit", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12")


def parse_train(s, datasets):
    out = {}
    for part in s.split(","):
        ds, _eq, looks = part.partition("=")
        lk = looks.split("+")
        if ds not in datasets or ds in out or not lk or any(x not in train_looks(ds) for x in lk) or len(set(lk)) != len(lk):
            raise SystemExit(f"--train {part}: a training dataset and its looks among {train_looks(ds) if ds in ORDER else ORDER}")
        out[ds] = lk
    if set(out) != set(datasets):
        raise SystemExit("--train names every training dataset once")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, default=R_TOP)
    ap.add_argument("--hold", default=None)
    ap.add_argument("--mres", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ce", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--teacher", default="gm")
    a = ap.parse_args(argv)
    out_path = Path(a.out)
    if "joint" not in out_path.name or (a.mres and "mres" not in out_path.name):
        raise SystemExit("--out must carry 'joint' (and 'mres' with --mres) in its file name")
    datasets = a.datasets.split(",")
    if len(datasets) < 2 or len(set(datasets)) != len(datasets) or any(ds not in ORDER for ds in datasets) \
            or datasets != sorted(datasets, key=ORDER.index):
        raise SystemExit(f"--datasets: two or more of {ORDER}, in that order")
    trains = parse_train(a.train, datasets)
    if a.m not in CL.MS:
        raise SystemExit(f"--m among {CL.MS}")
    hold = tuple(int(x) for x in a.hold.split("/")) if a.hold else None
    if hold and not (0 <= hold[0] < hold[1] and hold[1] >= 2):
        raise SystemExit("--hold h/F with 0 <= h < F, F >= 2")
    specs = {}
    for v in a.variants.split(","):
        sp = G3.parse("x4:" + v, ("x4",))
        if sp["pca"] or sp["K"] < a.m:
            raise SystemExit(f"{v}: no PCA code, and K >= M")
        sp["train"] = {ds: trains[ds] for ds in datasets}
        specs[v] = sp
    CL.check_pins()
    if AW.sha(Path(CL.__file__)) != PINS["anchor_clu"]:
        raise SystemExit("anchor_clu.py is not the pinned file")
    MR = CL.mres_install(a.mres)
    torch.set_num_threads(2)
    t0 = time.time()
    N = min(a.n, R_TOP)
    max_len = max(sp["max_len"] for sp in specs.values())

    # 1. every dataset's phrases and rows, in ORDER
    Q, part, src = [], {}, {}
    for ds in datasets:
        AW6.rebind(ds)
        vocab = CL.source_vocab()
        phi, phi_sha = CL._orig["phrase_table"](ds)
        Qd, pd, rec_sha, load_state = AG.load_looks(["x1", "select"] + trains[ds], max_len, t0)
        checks, voc = AW.anchor_tables(Qd)
        if list(voc.get("w2", []))[:N] != vocab[:N]:
            raise SystemExit(f"{ds}: the anchors' phrase strings are not the compact's")
        off = len(Q)
        Q.extend(Qd)
        for lk, rows in pd.items():
            part[(ds, lk)] = [off + i for i in rows]
        held = CL.held_fold(vocab, N, *hold) if hold else None
        src[ds] = {"vocab": vocab, "phi": phi, "held": held,
                   "info": {"phrase_table": phi_sha, "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA, "records": rec_sha,
                            "flag_checks": checks, "pruned_loader": load_state, "looks": {lk: len(r) for lk, r in pd.items()},
                            "held_phrases": None if held is None else int(held.sum())}}
        log(f"{ds}: {len(Qd)} rows, {time.time() - t0:.0f}s")
        del Qd

    # 2. the clusters: pooled phrase texts, held strings left out of every dataset
    X = []
    for ds in datasets:
        P = np.asarray(src[ds]["phi"][:N], dtype=np.float32)
        keep = np.linalg.norm(P, axis=1) > 1e-6
        if src[ds]["held"] is not None:
            keep &= ~src[ds]["held"][:N]
        X.append(P[keep])
    X = np.concatenate(X)
    if X.shape[0] < a.m:
        raise SystemExit(f"{X.shape[0]} phrases for {a.m} clusters")
    C = A16.kmeans_sph(X, a.m, seed=CL.SEED, iters=CL.ITERS).astype(np.float32)
    labs = {ds: CL.assign(src[ds]["phi"], C, N) for ds in datasets}
    T = CL.centroid_table(C)
    t_sha = hashlib.sha256(T.tobytes()).hexdigest()
    clu = {"M": a.m, "N": N, "seed": CL.SEED, "iters": CL.ITERS, "source": "+".join(datasets),
           "source_table": {ds: src[ds]["info"]["phrase_table"] for ds in datasets}, "held_excluded": hold is not None, "hold": a.hold,
           "mres": bool(a.mres), "centroid_table_sha256": t_sha, "C": C, "lab": labs, "pooled_phrases": int(X.shape[0])}
    sizes = {ds: np.bincount(labs[ds][labs[ds] >= 0], minlength=a.m).tolist() for ds in datasets}
    shared = int(sum(1 for c in range(a.m) if all(sizes[ds][c] > 0 for ds in datasets)))
    members = {c: {ds: [src[ds]["vocab"][r] for r in np.flatnonzero(labs[ds][:N] == c)[:4]] for ds in datasets} for c in range(a.m)}
    log(f"{a.m} clusters over {X.shape[0]} pooled phrases; {shared} hold phrases of every dataset")
    del X

    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    tr = [i for ds in datasets for lk in trains[ds] for i in part[(ds, lk)]]
    val = [i for ds in datasets for i in part[(ds, "select")]]
    zT_of = [None] * len(Q)
    if a.kd > 0:
        for i in tr:
            zT_of[i] = AW8.teacher_of(Q[i], a.teacher)
    B = {ds: part[(ds, "x1")][1::2] for ds in datasets}
    RB = {ds: AG.Reader(Q, B[ds], 20261002) for ds in datasets}
    res = {"look": "anchor_joint", "mode": "fit", "datasets": datasets, "train": trains, "script_sha256": AW.sha(Path(__file__)),
           "pins": PINS, "sources": {ds: src[ds]["info"] for ds in datasets}, "epochs": a.epochs, "lr": a.lr, "wd": a.wd,
           "kd": {"ce": a.ce, "kd": a.kd, "T": a.T, "teacher": a.teacher} if a.kd > 0 else None, "ce": a.ce, "hold": a.hold, "mres": bool(a.mres),
           "clusters": {k: v for k, v in clu.items() if k not in ("C", "lab")} | {"sizes": sizes, "shared_clusters": shared, "members": members},
           "train_rows": {ds: sum(len(part[(ds, lk)]) for lk in trains[ds]) for ds in datasets},
           "select_rows": {ds: len(part[(ds, "select")]) for ds in datasets}, "B": {ds: RB[ds].base() for ds in datasets}, "variants": {}}
    per_row = {}
    model_dir = out_path.parent / (out_path.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    def save():
        res["mres_doubled"] = dict(MR.STATS) if MR else None
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, **{f"B_rows_{ds}": np.asarray(B[ds]) for ds in datasets})

    # 3. per variant: types per dataset through its own cluster map, one fit on the union, reads per dataset
    for vi, (name, sp) in enumerate(specs.items()):
        t1 = time.time()
        K = sp["K"]
        TY = [None] * len(Q)
        alt = {ds: {} for ds in datasets}
        nt = None
        for ds in datasets:
            rows_ds = sorted({i for lk in ["x1", "select"] + trains[ds] for i in part[(ds, lk)]})
            nt, ty = AG.types_for(Q, sp, CL.cluster_map(labs[ds], K, src[ds]["held"]), rows_ds)
            for i in rows_ds:
                TY[i] = ty[i]
            alt[ds]["NR"] = AG.types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B[ds])[1]
            if hold:
                alt[ds]["REV"] = AG.types_for(Q, sp, CL.cluster_map(labs[ds], K), B[ds])[1]
        make = G3.make_for(sp, nt, T if sp["family"] != "free" else None)
        main_key = "MASK" if hold else "ID"
        hrows = {ds: AG.held_reach_rows(Q, B[ds], src[ds]["held"], N) for ds in datasets} if hold else {}
        seeds_out = {}
        m0 = None
        for sd in sp["seeds"]:
            t2 = time.time()
            model, reads, best_ep, curve, margin, by_margin = AW8.fit_read_kd(Q, TY, nt, tr, val, {f"B_{ds}": B[ds] for ds in datasets}, z_of, zT_of,
                                                                              make, a.epochs, sd, sp["rule"], a.lr, a.wd, a.ce, a.kd, a.T)
            if m0 is None:
                m0 = model
            rd = {}
            for ds in datasets:
                R = RB[ds]
                main_m = reads[f"B_{ds}"]
                e = {main_key: R.record(main_m)}
                per_row[f"{vi}_{sd}_{ds}_{main_key}"] = main_m
                for nm, TYa in alt[ds].items():
                    m_alt = AG.read(model, Q, TYa, nt, B[ds], z_of, margin)
                    e[nm] = R.record(m_alt)
                    e[f"{nm} - {main_key}"] = AW3.boot_pair(m_alt - main_m, R.W)
                    per_row[f"{vi}_{sd}_{ds}_{nm}"] = m_alt
                if hold:
                    hr = hrows[ds]
                    e["held_reach_rows"] = {"MASK": R.subset(main_m, hr)}
                    if hr:
                        pos = {i: j for j, i in enumerate(B[ds])}
                        hx = np.asarray([pos[i] for i in hr], dtype=np.int64)
                        e["held_reach_rows"]["REV"] = R.subset(per_row[f"{vi}_{sd}_{ds}_REV"], hr)
                        e["held_reach_rows"]["REV - MASK"] = AW3.boot_pair((per_row[f"{vi}_{sd}_{ds}_REV"] - main_m)[hx], R.W[:, hx])
                rd[ds] = e
            pt = model_dir / f"v{vi}_s{sd}.pt"
            G3.save_model(pt, model, sp, nt, margin, "+".join(datasets), [], t_sha,
                          {"hold": a.hold, "name": name, "seed": sd, "rdrop": None, "clusters": clu,
                           "joint": {"datasets": datasets, "train": trains, "lr": a.lr, "kd": a.kd, "ce": a.ce}})
            seeds_out[str(sd)] = {"best_epoch": best_ep, "curve": curve, "margin": None if margin is None else str(margin), "by_margin_select": by_margin,
                                  "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
                                  "model_file": str(pt.relative_to(ROOT)) if pt.is_relative_to(ROOT) else str(pt), "reads": rd,
                                  "seconds": round(time.time() - t2, 1)}
            log(f"{name}#{sd}: ep {best_ep}, " + "; ".join(f"{ds} " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in rd[ds].items()
                                                                                if isinstance(v, dict) and "rho (R@5, FC@5, hit@1)" in v) for ds in datasets))
        v = {**{k: x for k, x in sp.items() if k != "train"}, "tokens": nt, "params": int(sum(p.numel() for p in m0.parameters())),
             "params_used": AG.params_used(m0, sp), "types_per_row": round(float(np.mean([len(TY[i]) for i in tr])), 1), "seeds_read": seeds_out}
        if hold:
            v["held_reach_rows"] = {ds: len(hrows[ds]) for ds in datasets}
        v["seconds"] = round(time.time() - t1, 1)
        res["variants"][name] = v
        save()
        del TY, alt
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


if __name__ == "__main__":
    main()
