"""ENC E2N (docs/ENC_ENCODER_COMPARISON.md section 15, declared 20 Nov 2026 06:40 IST before any E2N number): node
memory on HotpotQA. A document's vector moves toward the questions that reached it in training, and the first stage is
read again on HotpotQA's s1eval.

    python outputs/mp_unified/zenc2n.py selftest
    python outputs/mp_unified/zenc2n.py run --threads 8       (host CPU)
    python outputs/mp_unified/zenc2n.py report

A document d with a feedback set Q_d becomes unit(d + lambda unit(mean of Q_d's question vectors)); others are unchanged.
Forms (15.2): gold (train-split questions with d among their golds) and self (train-split questions whose raw dense top
1 is d). s1sel's questions feed nothing back. Encoders mpnet (decides) and frozen (beside). lambda by s1sel dense R@5.
Reads: the raw top 2,000 over the whole index, with every changed document rescored. Writes only under outputs/enc/e2n/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)
import zenc2m as M  # noqa: E402
import zenc2s as S  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "enc" / "e2n"
E0 = ROOT / "outputs" / "enc" / "e0"
STAGE = "ENC E2N"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 15"
DS = "hotpotqa"
LAMBDAS = (0.25, 0.5, 1.0, 2.0)
FORMS = ("gold", "self")
ENCS = ("mpnet", "frozen")
K2 = 2000


def log(msg):
    print(f"[zenc2n {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc2n: {msg}")


def code_shas():
    return {f"outputs/mp_unified/{Path(p).name}": E.sha256(Path(p)) for p in (__file__, E.__file__, M.__file__, S.__file__)}


# ── memory ───────────────────────────────────────────────────────────────────

def feedback_pairs(form, keep, golds, top1=None):
    """(question index into keep, document) pairs of the form's feedback."""
    if form == "gold":
        qi = np.concatenate([np.full(golds[r].size, i, np.int64) for i, r in enumerate(keep)] + [np.zeros(0, np.int64)])
        di = np.concatenate([golds[r] for r in keep] + [np.zeros(0, np.int64)])
        return qi, di
    return np.arange(keep.size, dtype=np.int64), np.asarray(top1, np.int64)


def memory_dirs(Qk, qi, di):
    """(changed documents, sorted; unit(mean of their questions' vectors)) for the pairs."""
    from scipy import sparse
    Md, inv = np.unique(di, return_inverse=True)
    A = sparse.csr_matrix((np.ones(qi.size, np.float32), (inv, qi)), shape=(Md.size, Qk.shape[0]))
    return Md, M.unit(np.asarray(A @ Qk, np.float32))


def updated(Draw, Mdir, lam):
    return M.unit(Draw + np.float32(lam) * Mdir)


def merged_top(raw_ids, raw_sc, Qc, Md, U, k, qb=512):
    """Top k per question over (raw documents outside Md, with their raw scores) and (Md, rescored as U). exact[q] says
    whether at least k unchanged documents sat in the raw list, so the top k is exact for any k up to that."""
    nq = Qc.shape[0]
    out = np.empty((nq, k), np.int64)
    n_unch = np.empty(nq, np.int64)
    km = min(k, Md.size)
    for q0 in range(0, nq, qb):
        q1 = min(nq, q0 + qb)
        Sm = Qc[q0:q1] @ U.T
        part = np.argpartition(-Sm, km - 1, axis=1)[:, :km]
        ps = np.take_along_axis(Sm, part, 1)
        unch = ~np.isin(raw_ids[q0:q1], Md)
        for i in range(q1 - q0):
            q = q0 + i
            ci = np.concatenate([raw_ids[q][unch[i]], Md[part[i]]])
            cs = np.concatenate([raw_sc[q][unch[i]], ps[i]])
            o = np.lexsort((ci, -cs))[:k]
            out[q] = ci[o]
            n_unch[q] = int(unch[i].sum())
    return out, n_unch


def save_read(name, c, h, top):
    d = OUT / "reads"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}__{DS}.npz"
    np.savez(p, ids=np.asarray(c["ids"]), rows=c["rows"], gold_total=c["gt"], top20=top[:, :20].astype(np.int32),
             **{f"h_{k}": v.astype(np.int16) for k, v in h.items()})
    return {"npz": p.name, "sha256": E.sha256(p), "recall": {k: round(S.recall_share(v, c["gt"]), 6) for k, v in h.items()}}


def cmd_run(a):
    root = E.served_root()
    t0 = time.time()
    if (OUT / "run.json").exists():
        stop("outputs/enc/e2n/run.json exists: the stage runs once")
    golds, qids = S.gold_rows(root, DS)
    cs, ce = S.sel_carve(DS, golds, qids), M.carve(root, DS)
    tr = M.split_rows(root, DS)["train"]
    keep = np.setdiff1d(tr, cs["rows"])
    if np.isin(ce["rows"], keep).any():
        stop("an s1eval question is among the feedback questions")
    gold_docs = np.unique(np.concatenate([golds[r] for r in keep]))
    seen = np.asarray([np.isin(ce["flat"][ce["off"][q]:ce["off"][q + 1]], gold_docs).any()
                       for q in range(len(ce["ids"]))])
    rec = {"stage": STAGE, "declared_in": DECLARED, "feedback_questions": int(keep.size), "train_questions": int(tr.size),
           "s1sel_left_out": int(np.isin(cs["rows"], tr).sum()), "s1eval_seen_share": round(float(seen.mean()), 6),
           "encs": {}, "g1": {}, "reads": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "seen.npy", seen)
    for enc in ENCS:
        t = time.time()
        paths = E.vec_shards(root, enc, DS, "docs")
        Qk = S.rows_of(root, enc, DS, "queries", keep)
        Qs = S.rows_of(root, enc, DS, "queries", cs["rows"])
        Qe = S.rows_of(root, enc, DS, "queries", ce["rows"])
        rs_i, rs_s, _ = E.search(Qs, paths, K2)
        re_i, re_s, _ = E.search(Qe, paths, K2)
        rec["g1"][enc] = S.g1_check(S.read_of(re_i[:, :E.K], ce), enc, DS, ce)
        log(f"{enc}: raw reads done ({round(time.time() - t, 1)} s), G1 {rec['g1'][enc]['pass']}")
        if enc == "frozen":
            top1 = np.asarray(np.load(root / DS / "retrieval_cache" / "dense_top1000.npz", mmap_mode="r")["ids"][keep][:, 0],
                              np.int64)
        else:
            top1 = E.search(Qk, paths, 1)[0][:, 0]
        er = {"self_top1_is_gold_share": round(float(np.mean([np.isin(top1[i], golds[r]) for i, r in enumerate(keep)])), 6)}
        for form in FORMS:
            qi, di = feedback_pairs(form, keep, golds, top1)
            Md, Mdir = memory_dirs(Qk, qi, di)
            Draw = S.rows_of(root, enc, DS, "docs", Md)
            curve = []
            for lam in LAMBDAS:
                U = updated(Draw, Mdir, lam)
                top, n_unch = merged_top(rs_i, rs_s, Qs, Md, U, 5)
                curve.append({"lambda": lam, "s1sel_r5": S.r5_of(top, cs), "exact5_share": round(float(np.mean(n_unch >= 5)), 6)})
                log(f"{enc} {form} s1sel lambda={lam}: R@5 {round(curve[-1]['s1sel_r5'], 6)}")
            best = S.pick(curve, key=lambda x: x["s1sel_r5"], order=lambda x: (x["lambda"],))
            U = updated(Draw, Mdir, best["lambda"])
            top, n_unch = merged_top(re_i, re_s, Qe, Md, U, E.K)
            name = f"{enc}-{form}"
            rec["reads"][name] = save_read(name, ce, S.read_of(top, ce), top)
            rec["reads"][name]["exact1000_share"] = round(float(np.mean(n_unch >= E.K)), 6)
            er[form] = {"changed_docs": int(Md.size), "pairs": int(qi.size), "curve": curve, "chosen": best["lambda"]}
            log(f"{enc} {form} s1eval (lambda={best['lambda']}): {json.dumps(rec['reads'][name]['recall'])}")
        er["seconds"] = round(time.time() - t, 1)
        rec["encs"][enc] = er
    rec.update(threads=E.THREADS, code_sha256=code_shas(), seconds=round(time.time() - t0, 1),
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    E.wjson(OUT / "run.json", rec)
    log(f"run done in {rec['seconds']} s")
    return 0


# ── the report ───────────────────────────────────────────────────────────────

def cmd_report(a):
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    if not all(v["pass"] for v in run["g1"].values()):
        stop(f"G1 failed: {json.dumps(run['g1'])}")
    seen = np.load(OUT / "seen.npy")
    zf, zm = np.load(E0 / "frozen" / f"{DS}.npz"), np.load(E0 / "mpnet" / f"{DS}.npz")
    e0gap = M.gap(M.per_q(zm, "d5"), M.per_q(zf, "d5"))["pts"]
    rep = {"stage": STAGE, "declared_in": DECLARED, "g1": run["g1"], "e0_gap_mpnet_to_frozen": e0gap, "cells": {},
           "calls": {}, "s1eval_seen_share": run["s1eval_seen_share"]}
    lines = ["# ENC E2N: node memory on HotpotQA (report)", "",
             f"Written by `outputs/mp_unified/zenc2n.py report` under {DECLARED}. R@5 x100 on HotpotQA's s1eval "
             f"({len(seen)} questions; {round(100 * float(seen.mean()), 1)}% have a gold among the feedback golds). Won "
             "back = the read minus E0's raw read of the same encoder, points, paired 95% bootstrap (2,000, seed 0).", "",
             "| encoder | form | lambda | R@5 | won back | won back, seen | won back, not seen | gap to frozen | "
             "R@1000 exact share |", "|---|---|---|---|---|---|---|---|---|"]
    for enc in ENCS:
        zb = zm if enc == "mpnet" else zf
        for form in FORMS:
            name = f"{enc}-{form}"
            z = np.load(OUT / "reads" / f"{name}__{DS}.npz")
            if not S.same_ids(z, zb, zf):
                stop(f"{name}: the reads hold other questions")
            x, b = M.per_q(z, "d5"), M.per_q(zb, "d5")
            c = {"r5": round(100 * float(x.mean()), 2), "won_back": M.gap(x, b),
                 "seen": M.gap(x[seen], b[seen]) if seen.any() else None,
                 "not_seen": M.gap(x[~seen], b[~seen]) if (~seen).any() else None,
                 "gap": {k: M.gap(M.per_q(z, k), M.per_q(zf, k)) for k in ("d5", "d20", "d1000", "rrf5")},
                 "lambda": run["encs"][enc][form]["chosen"], "exact1000_share": run["reads"][name]["exact1000_share"]}
            if enc == "mpnet":
                c["share_of_gap_won"] = round(c["won_back"]["pts"] / -e0gap, 4) if e0gap < 0 else None
                rep["calls"][form] = {"ABOVE": "HELPS", "LEVEL": "LEVEL", "BELOW": "HURTS"}[c["won_back"]["call"]]
            rep["cells"][name] = c
            lines.append(f"| {enc} | {form} | {c['lambda']} | {c['r5']} | {c['won_back']['pts']} {c['won_back']['ci']} | "
                         f"{c['seen']['pts'] if c['seen'] else '-'} | {c['not_seen']['pts'] if c['not_seen'] else '-'} | "
                         f"{c['gap']['d5']['pts']} {c['gap']['d5']['ci']} | {c['exact1000_share']} |")
    lines += ["", f"Calls (15.4, mpnet): {json.dumps(rep['calls'])}", "",
              f"E0's gap, mpnet to frozen: {e0gap} points. Self feedback's top 1 is a gold for "
              f"{json.dumps({e: run['encs'][e]['self_top1_is_gold_share'] for e in ENCS})} of the feedback questions."]
    rep["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    E.wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {json.dumps(rep['calls'])}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    rng = np.random.default_rng(5)
    nd, dim, nq = 300, 16, 40
    D = M.unit(rng.standard_normal((nd, dim)).astype(np.float32))
    Q = M.unit(rng.standard_normal((nq, dim)).astype(np.float32))
    # memory directions: the unit mean of each changed document's questions
    Qk = M.unit(rng.standard_normal((25, dim)).astype(np.float32))
    qi, di = np.asarray([0, 1, 2, 3, 3]), np.asarray([7, 7, 9, 9, 4])
    Md, Mdir = memory_dirs(Qk, qi, di)
    assert Md.tolist() == [4, 7, 9]
    assert np.allclose(Mdir[1], M.unit((Qk[0] + Qk[1])[None])[0], atol=1e-6)
    assert np.allclose(Mdir[2], M.unit((Qk[2] + Qk[3])[None])[0], atol=1e-6)
    golds = [np.asarray([5, 8]), np.asarray([1])]
    q2, d2 = feedback_pairs("gold", np.asarray([0, 1]), golds)
    assert q2.tolist() == [0, 0, 1] and d2.tolist() == [5, 8, 1]
    # the merge equals a full search over the updated corpus whenever enough unchanged documents sit in the raw list
    Md = np.sort(rng.choice(nd, 30, replace=False))
    U = updated(D[Md], M.unit(rng.standard_normal((30, dim)).astype(np.float32)), 1.0)
    Dfull = D.copy()
    Dfull[Md] = U
    raw_i, raw_s = S.topk(lambda q0, q1: Q[q0:q1] @ D.T, nq, 120)
    for k in (5, 20, 60):
        top, n_unch = merged_top(raw_i, raw_s, Q, Md, U, k, qb=7)
        ref, _ = S.topk(lambda q0, q1: Q[q0:q1] @ Dfull.T, nq, k)
        ok = n_unch >= k
        assert ok.all() and np.array_equal(top[ok], ref[ok]), k
    # the call labels
    assert {"ABOVE": "HELPS", "LEVEL": "LEVEL", "BELOW": "HURTS"}["BELOW"] == "HURTS"
    print("zenc2n self-test: OK", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc2n.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--threads", default="8")
    sub.add_parser("report")
    p = sub.add_parser("selftest")
    p.add_argument("--threads", default="8")
    a = ap.parse_args()
    return {"run": cmd_run, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
