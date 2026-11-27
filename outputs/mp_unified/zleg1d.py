"""LEG1 part B on MetaQA (docs/LEG1_LEG_NECESSITY.md section 11, amendment LEG1-A3): the stored C0 vectors, their caps and the link-mixed vectors on the hop-rich dataset.

    python -I -B outputs/mp_unified/zleg1d.py selftest
    python -I -B outputs/mp_unified/zleg1d.py vectors --ds metaqa [--device cpu] [--threads N]
        # host row: the record's pooling (arm l0) of every node and every eval question as CSR shards under outputs/leg1/vec/<ds>/ (resumable); the sequence share of the top 5
        # against the record's list is stored as information (meta.json), not as a gate
    python -I -B outputs/mp_unified/zleg1d.py read --ds metaqa
        # host row (needs the e3v lists and the filed links): G-B0c (section 9's gate, directly), then zleg1c's caps (B1, B2), then the 24-arm mix (B3) with the parity split;
        # writes outputs/leg1/gate_gb0c_<ds>.json, caps_<ds>.json, mix_<ds>.json
    python -I -B outputs/mp_unified/zleg1d.py legs --ds metaqa
        # LEG1-A4 (section 12): part A on MetaQA (zleg1's legs, unchanged, with the host's filed first-stage record as G-S1's value); writes outputs/leg1/legs_<ds>.json

Run from the workspace root. zspl1.py, zspl1l.py, zleg1.py, zleg1b.py, zleg1c.py and zul2p.py (load_host) are imported and edited nowhere.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import zleg1b as B  # noqa: E402
import zleg1c as C  # noqa: E402
import zspl1 as S  # noqa: E402

DOC_A3 = "docs/LEG1_LEG_NECESSITY.md section 11 (LEG1-A3)"
OUT = C.OUT
DATASETS = ("metaqa",)                      # LEG1-A3: MetaQA only; any other dataset by its own amendment
LEGS_DATASETS = ("metaqa",)                 # LEG1-A4: part A on MetaQA only
GRAPHS3 = ("S", "R", "K", "SRK")            # LEG1-A3: the knowledge-base triplets, the rule links CN_all, the e3v kNN pairs Ke, their union
SIZES = ("full", f"cap{C.MIX_CAP}")
log, stop = S.log, S.stop


def graph_keys(F):
    """The four graphs of LEG1-A3 as sorted unique pair keys (a * n + b)."""
    s, r, k = (np.unique(np.asarray(F[x], np.int64)) for x in ("S", "CN_all", "Ke"))
    return {"S": s, "R": r, "K": k, "SRK": np.union1d(np.union1d(s, r), k)}


def parity_split(nq):
    """LEG1-A3: the dev half is the even question index, the held-out half the odd one."""
    dev = (np.arange(nq) % 2) == 0
    return dev, ~dev


def cmd_vectors(a):
    import scipy.sparse as sp
    import zworld
    t0 = time.time()
    if a.ds not in DATASETS:
        stop("LEG1-A3 declares MetaQA only; other datasets by their own amendment")
    device = a.device or "cpu"
    torch, tok, models, tfv = B.L.load_models(device, ["l0"])
    if a.threads:
        torch.set_num_threads(a.threads)
    w = zworld.load(a.ds)
    rows, _ = S.eval_rows(a.ds, w)
    qt = S.question_texts(w, rows)
    texts = [t for _, t in w.iter_nodes()]
    d = B.OUT / a.ds
    d.mkdir(parents=True, exist_ok=True)
    log(f"{a.ds}: {len(texts):,} nodes, {len(qt):,} questions, device {device}, {torch.get_num_threads()} threads")
    tq = B.encode_shards(sp, torch, tok, models, device, qt, d, "q", B.QBLOCK)
    td = B.encode_shards(sp, torch, tok, models, device, texts, d, "d", S.BLOCK)
    Qm, Dm = B.load_shards(sp, d, "q", len(qt)), B.load_shards(sp, d, "d", len(texts))
    rec = np.asarray(np.load(w.root / "retrieval_cache" / "splade_top1000.npz", mmap_mode="r")["ids"][rows][:, :5], np.int64)
    got = B.top5(sp, Qm, Dm)
    share = float(np.mean(np.all(got == rec, axis=1)))
    set_share = float(np.mean(np.all(np.sort(got, axis=1) == np.sort(rec, axis=1), axis=1)))
    meta = {"stage": "LEG1 part B vectors (A3)", "declared_in": DOC_A3, "ds": a.ds, "nodes": len(texts), "questions": len(qt), "device": device, "threads": torch.get_num_threads(),
            "doc_nnz_total": int(Dm.nnz), "doc_nnz_mean": round(Dm.nnz / len(texts), 2), "query_nnz_mean": round(Qm.nnz / len(qt), 2),
            "gate_gb0": {"top5_same_sequence_share": round(share, 5), "top5_same_set_share": round(set_share, 5), "min_share": B.GATE_SHARE, "pass": share >= B.GATE_SHARE, "information_only": True},
            "encode_seconds_this_run": {"questions": round(tq, 1), "nodes": round(td, 1)}, "seconds": round(time.time() - t0, 1), "versions": {"torch": torch.__version__, "transformers": tfv},
            "sha256": {"doc_shard0": S.sha256(B.shard_path(d, "d", 0)), "q_shard0": S.sha256(B.shard_path(d, "q", 0))}}
    S.wjson(d / "meta.json", meta)
    log(f"sequence share {share:.4f}, set share {set_share:.4f} (information only; the gate is G-B0c in the read row)")
    log(f"done {a.ds}: doc nnz mean {meta['doc_nnz_mean']}, query nnz mean {meta['query_nnz_mean']}, {meta['seconds']} s")


def gate_read(a):
    """G-B0c (LEG1-A2) applied directly (LEG1-A3): the stored list against the record's C0 list on the leg's R@20 and the seeds."""
    import scipy.sparse as sp
    import zenc as E
    t0 = time.time()
    d = B.OUT / a.ds
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").is_file() else None
    if meta is None:
        stop(f"{a.ds}: no stored vectors")
    c = S.context(E, a.ds)
    n, dense = c["w"].n, c["dense"]
    Qm, Dm = B.load_shards(sp, d, "q", len(c["rows"])), B.load_shards(sp, d, "d", n)
    mine = C.topk(sp, Qm, Dm, 1000)
    r_mine = C.reads_of_list(E, dense, mine, c["flat"], c["off"], c["gt"], n)
    r_rec = C.reads_of_list(E, dense, c["c0"], c["flat"], c["off"], c["gt"], n)
    res = {"stage": "LEG1 G-B0c", "declared_in": DOC_A3, "ds": a.ds, "questions": int(len(c["rows"])), "nodes": int(n), "gate_gb0": meta["gate_gb0"], "line_pts": C.GATE_C_PTS, "cells": {},
           "reference_reads_stored": {k: round(float(v.mean() * 100), 3) for k, v in r_mine.items()}}
    for k in C.GATE_C_CELLS:
        dm, lo, hi = S.boot(r_mine[k] - r_rec[k])
        res["cells"][k] = {"record": round(float(r_rec[k].mean() * 100), 3), "stored": round(float(r_mine[k].mean() * 100), 3), "diff": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3)}
        log(f"  {a.ds} {k}: record {res['cells'][k]['record']:.3f}, stored {res['cells'][k]['stored']:.3f}, stored - record {dm:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    res["pass"] = C.gate_c_call([res["cells"][k]["lo"] for k in C.GATE_C_CELLS])
    res["seconds"] = round(time.time() - t0, 1)
    S.wjson(OUT / f"gate_gb0c_{a.ds}.json", res)
    log(f"G-B0c on {a.ds}: {'PASS' if res['pass'] else 'FAIL, part B is closed on this dataset'}")
    if not res["pass"]:
        stop(f"G-B0c failed on {a.ds}")


def mix_read(a):
    """B3 on MetaQA (LEG1-A3): 24 arms, the even index picks, the odd index calls."""
    import scipy.sparse as sp
    import zenc as E
    import zul2p
    t0 = time.time()
    d = B.OUT / a.ds
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    gf = OUT / f"gate_gb0c_{a.ds}.json"
    if not gf.is_file() or not json.loads(gf.read_text(encoding="utf-8")).get("pass"):
        stop(f"{a.ds}: G-B0c has not passed: nothing is read")
    c = S.context(E, a.ds)
    n, dense, nq = c["w"].n, c["dense"], len(c["rows"])
    D = zul2p.load_host(a.ds, True)
    if D.n != n:
        stop(f"the links' node count {D.n} is not the vectors' {n}")
    keys = graph_keys(D.F)
    if keys["S"].size == 0:
        stop(f"{a.ds}: the S graph is empty; the amendment names the knowledge-base triplets")
    Qm, Dm = B.load_shards(sp, d, "q", nq), B.load_shards(sp, d, "d", n)
    dev, held = parity_split(nq)
    ref_leg = C.topk(sp, Qm, Dm, 1000)
    ref_r = C.reads_of_list(E, dense, ref_leg, c["flat"], c["off"], c["gt"], n)
    cells = [k for k in ("rrf5", "rrf20", "seeds", "leg20", "leg1000") if k in ref_r]
    res = {"stage": "LEG1 part B3 mix (A3)", "declared_in": DOC_A3, "ds": a.ds, "questions": nq, "dev": int(dev.sum()), "held": int(held.sum()), "nodes": int(n),
           "links": {k: int(v.size) for k, v in keys.items()}, "gate_gb0": meta["gate_gb0"], "gate_gb0c_file": gf.name,
           "reference": {"dev": {k: round(float(ref_r[k][dev].mean() * 100), 3) for k in cells}, "held": {k: round(float(ref_r[k][held].mean() * 100), 3) for k in cells}, "doc_nnz_mean": round(Dm.nnz / n, 2)},
           "arms": {}}
    dev_seeds, nnzs, store = {}, {}, {}
    for g in GRAPHS3:
        A = C.adjacency(sp, keys[g], n)
        linked = int(np.sum(np.asarray(A.sum(axis=1)).ravel() > 0))
        for al in C.ALPHAS:
            Dx = C.mix(sp, Dm, A, al)
            for size in SIZES:
                name = f"{g}_a{al}_{size}"
                D2 = Dx if size == "full" else C.cap_rows(sp, Dx, C.MIX_CAP)
                leg = C.topk(sp, Qm, D2, 1000)
                r = C.reads_of_list(E, dense, leg, c["flat"], c["off"], c["gt"], n)
                rc, ls = C.enter_leave(ref_leg, leg, c["flat"], c["off"])
                store[name] = r
                dev_seeds[name], nnzs[name] = float(r["seeds"][dev].mean()), D2.nnz / n
                res["arms"][name] = {"graph": g, "alpha": al, "size": size, "doc_nnz_mean": round(D2.nnz / n, 2), "nodes_changed": linked,
                                     "dev": {k: round(float(r[k][dev].mean() * 100), 3) for k in cells}, "held": {k: round(float(r[k][held].mean() * 100), 3) for k in cells},
                                     "leg20_golds_recovered_per_question": round(float(rc.mean()), 4), "leg20_golds_lost_per_question": round(float(ls.mean()), 4)}
                log(f"  {name}: nnz {D2.nnz / n:.0f}, dev seeds {dev_seeds[name] * 100:.2f}, held seeds {res['arms'][name]['held']['seeds']:.2f} (ref {res['reference']['held']['seeds']:.2f}), leg20 recovered {rc.mean():.3f} lost {ls.mean():.3f}")
            del Dx
    pk = C.pick_arm(dev_seeds, nnzs)
    diffs = {}
    for k in cells:
        dm, lo, hi = S.boot((store[pk][k] - ref_r[k])[held])
        diffs[k] = {"diff": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3), "call": S.call_of(lo)}
    call = C.mix_call(diffs["seeds"]["lo"], [diffs[k]["lo"] for k in cells if k != "seeds"])
    res["pick"] = {"arm": pk, "dev_seeds": round(dev_seeds[pk] * 100, 3), "held_vs_reference": diffs, "call": call, "doc_nnz_ratio_to_reference": round(nnzs[pk] / (Dm.nnz / n), 3)}
    res["seconds"] = round(time.time() - t0, 1)
    S.wjson(OUT / f"mix_{a.ds}.json", res)
    log(f"pick {pk}: held-out seeds {diffs['seeds']['diff']:+.2f} [{diffs['seeds']['lo']:+.2f}, {diffs['seeds']['hi']:+.2f}] -> {call}")


def filed_rrf5(ds, root=None):
    """The host's filed first-stage record of a dataset (outputs/enc/e0/e3v/<ds>.json): its fusion R@5, as a fraction (LEG1-A4's G-S1 value)."""
    p = Path(root if root is not None else S.WS / "outputs" / "enc" / "e0" / "e3v") / f"{ds}.json"
    if not p.is_file():
        stop(f"{ds}: the filed first-stage record {p.name} is not here")
    return float(json.loads(p.read_text(encoding="utf-8"))["recall"]["rrf5"])


def cmd_legs(a):
    """Part A on MetaQA (LEG1-A4): zleg1.cmd_legs unchanged, its context loader wrapped to supply the filed rrf5 where zspl1.context has none."""
    import zleg1 as Z
    if a.ds not in LEGS_DATASETS:
        stop("LEG1-A4 declares MetaQA only; other datasets by their own amendment")
    orig = S.context

    def ctx(E, ds):
        c = orig(E, ds)
        if c["e0_rrf5"] is None:
            c["e0_rrf5"] = filed_rrf5(ds)
        return c

    S.context = ctx
    try:
        Z.cmd_legs(a)
    finally:
        S.context = orig


def cmd_read(a):
    if a.ds not in DATASETS:
        stop("LEG1-A3 declares MetaQA only; other datasets by their own amendment")
    gate_read(a)
    C.cmd_caps(a)
    mix_read(a)


def selftest():
    bad = 0

    def chk(name, ok):
        nonlocal bad
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    chk("MetaQA only", DATASETS == ("metaqa",))
    n = 10
    F = {"S": np.array([1 * n + 2, 1 * n + 2, 3 * n + 4]), "CN_all": np.array([3 * n + 4, 5 * n + 6]), "Ke": np.array([7 * n + 8])}
    g = graph_keys(F)
    chk("the graphs: S 2 unique pairs, R 2, K 1, the union 4", [g[k].size for k in GRAPHS3] == [2, 2, 1, 4] and g["SRK"].tolist() == sorted({12, 34, 56, 78}))
    dev, held = parity_split(5)
    chk("parity split: even index dev, odd held, no overlap, together all", dev.tolist() == [True, False, True, False, True] and bool(np.all(dev ^ held)) and int(dev.sum()) == 3)
    chk("arms: 4 graphs x 3 weights x 2 sizes = 24", len(GRAPHS3) * len(C.ALPHAS) * len(SIZES) == 24)
    chk("the gate line is section 9's -0.5 on the leg's R@20 and the seeds", C.GATE_C_PTS == -0.5 and C.GATE_C_CELLS == ("leg20", "seeds"))
    chk("the cap and the not-worse line are section 7's", C.MIX_CAP == 128 and C.NOT_WORSE == -1.0)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "toy.json").write_text(json.dumps({"recall": {"rrf5": 0.004454, "d5": 0.1}}), encoding="utf-8")
        chk("the filed rrf5 is read from the record's recall block", filed_rrf5("toy", td) == 0.004454)
    chk("part A is declared for MetaQA only", LEGS_DATASETS == ("metaqa",))
    if bad:
        stop(f"selftest: {bad} check(s) failed")
    log("selftest PASS (8 checks)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    v = sub.add_parser("vectors")
    v.add_argument("--ds", required=True)
    v.add_argument("--device", default=None)
    v.add_argument("--threads", type=int, default=0)
    v.set_defaults(fn=cmd_vectors)
    r = sub.add_parser("read")
    r.add_argument("--ds", required=True)
    r.set_defaults(fn=cmd_read)
    lg = sub.add_parser("legs")
    lg.add_argument("--ds", required=True)
    lg.set_defaults(fn=cmd_legs)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
