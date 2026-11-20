"""ENC E5 (docs/ENC_ENCODER_COMPARISON.md section 21): how small e3v's document vectors can be stored, on all six
datasets, before the first stage moves. Declared before any E5 number. Reads only; nothing is trained on the labels.

Forms (documents; questions stay float32 and get the same projection):
  fp16             e3v as stored (FS: must reproduce E0/E4's e3v read exactly)
  int8             one scale per dimension = max |x| over the dataset's documents / 127
  pca768-own       centred on the dataset's documents, top 768 principal directions of its documents, unit length
  pca768-own+int8  both
  pca768-loo       the same, but the mean and directions come from the OTHER five datasets' documents (equal weight per
                   dataset): a corpus the projection never saw, as a new graph would be
  pca768-loo+int8  both

    python outputs/mp_unified/zenc5.py moments --ds DS [--threads N]   pass 1: sums, second moments, max |x| per dim
    python outputs/mp_unified/zenc5.py read --ds DS [--threads N]      pass 2 (scales) + pass 3 (all forms, one search)
    python outputs/mp_unified/zenc5.py report
    python outputs/mp_unified/zenc5.py selftest [--real]           --real: SQuAD fp16 through this search = its E0 read
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import zenc as E  # noqa: E402  (reads --threads from sys.argv before numpy loads)
import zenc3v as V  # noqa: E402  (registers e3v)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ENC = "e3v"
SIX = E.SIX
KP = 768
FORMS = ("fp16", "int8", "pca768-own", "pca768-own+int8", "pca768-loo", "pca768-loo+int8")
BYTES = {"fp16": 2 * V.DIM, "int8": V.DIM, "pca768-own": 2 * KP, "pca768-own+int8": KP, "pca768-loo": 2 * KP,
         "pca768-loo+int8": KP}
FROZEN_BYTES = 2 * 1536
OUT = E.OUT / "e5"
BOOT, SEED = 2000, 0
FREE_PTS = -0.30          # 21.4: a FREE form's point gaps are all >= -0.30 and its intervals LEVEL or ABOVE
KEYS = ("d5", "d1000", "rrf5")


def stop(msg):
    E.stop(f"E5: {msg}")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def shards(ds):
    return E.vec_shards(E.served_root(), ENC, ds, "docs")


def cmd_moments(a):
    t0 = time.time()
    S1 = np.zeros(V.DIM, np.float64)
    S2 = np.zeros((V.DIM, V.DIM), np.float64)
    mx = np.zeros(V.DIM, np.float32)
    n = 0
    for p in shards(a.ds):
        D = np.asarray(np.load(p, mmap_mode="r"), np.float32)
        S1 += D.sum(0, dtype=np.float64)
        S2 += (D.T @ D).astype(np.float64)
        mx = np.maximum(mx, np.abs(D).max(0))
        n += D.shape[0]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"moments_{a.ds}.npz", n=n, s1=S1, s2=S2, maxabs=mx)
    E.wjson(OUT / f"moments_{a.ds}.json", {"stage": "ENC E5: moments", "dataset": a.ds, "rows": n,
                                          "npz_sha256": E.sha256(OUT / f"moments_{a.ds}.npz"),
                                          "code_sha256": code_sha(), "seconds": round(time.time() - t0, 1), "utc": utc()})
    print(f"moments {a.ds}: {n:,} rows in {time.time() - t0:.0f} s", flush=True)
    return 0


def mom(ds):
    z = np.load(OUT / f"moments_{ds}.npz")
    n = int(z["n"])
    return z["s1"] / n, z["s2"] / n, z["maxabs"].astype(np.float32)


def pca_fit(sets):
    """Equal weight per dataset: mu = mean of the means, C = mean of the second moments - mu mu^T."""
    ms = [mom(d) for d in sets]
    mu = np.mean([m[0] for m in ms], 0)
    C = np.mean([m[1] for m in ms], 0) - np.outer(mu, mu)
    w, Vv = np.linalg.eigh(C)
    o = np.argsort(-w)
    return mu.astype(np.float32), Vv[:, o[:KP]].astype(np.float32), float(w[o[:KP]].sum() / w.sum())


def proj(X, mu, P):
    Y = (X - mu) @ P
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def quant(D, scale):
    return np.round(D / scale).clip(-127, 127).astype(np.int8).astype(np.float32) * scale


class Top:
    """zenc.search's merge (exact float32 inner product, top k, ties to the lower id) over arrays, not files."""

    def __init__(self, nq, k=E.K):
        self.k = k
        self.bs = np.full((nq, k), -np.inf, np.float32)
        self.bi = np.full((nq, k), np.iinfo(np.int64).max, np.int64)

    def add(self, Q, D, off):
        n = D.shape[0]
        for q0 in range(0, Q.shape[0], E.QB):
            q1 = min(Q.shape[0], q0 + E.QB)
            S = Q[q0:q1] @ D.T
            kk = min(self.k, n)
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            cs = np.concatenate([self.bs[q0:q1], np.take_along_axis(S, part, 1)], 1)
            ci = np.concatenate([self.bi[q0:q1], part.astype(np.int64) + off], 1)
            keep = np.argpartition(-cs, self.k - 1, axis=1)[:, :self.k]
            self.bs[q0:q1] = np.take_along_axis(cs, keep, 1)
            self.bi[q0:q1] = np.take_along_axis(ci, keep, 1)

    def result(self):
        o = np.lexsort((self.bi, -self.bs), axis=1)
        return np.take_along_axis(self.bi, o, 1)


def carve(ds):
    root = E.served_root()
    s1 = E.s1eval(ds)
    ids, gt = s1["ids"], s1["gold_total"]
    qids = json.loads((root / ds / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    at = {q: j for j, q in enumerate(qids)}
    rows = np.asarray([at[q] for q in ids], np.int64)
    flat, off = E.gold_sets(root, ds, ids, gt)
    qp = E.vec_shards(root, ENC, ds, "queries")
    Q = np.empty((rows.size, V.DIM), np.float32)
    for s in np.unique(rows // E.SHARD):
        A = np.load(qp[int(s)], mmap_mode="r")
        m = rows // E.SHARD == s
        Q[m] = A[rows[m] - s * E.SHARD]
    spl = np.asarray(np.load(root / ds / "retrieval_cache" / "splade_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)
    return ids, gt, flat, off, Q, spl


def cmd_read(a):
    t0 = time.time()
    ds = a.ds
    ids, gt, flat, off, Q, spl = carve(ds)
    fits = {"own": pca_fit([ds]), "loo": pca_fit([d for d in SIX if d != ds])}
    scale8 = np.where(mom(ds)[2] > 0, mom(ds)[2] / 127.0, 1.0).astype(np.float32)
    # pass 2: the int8 scales of the projected documents
    pmax = {h: np.zeros(KP, np.float32) for h in fits}
    for p in shards(ds):
        D = np.asarray(np.load(p, mmap_mode="r"), np.float32)
        for h, (mu, P, _) in fits.items():
            pmax[h] = np.maximum(pmax[h], np.abs(proj(D, mu, P)).max(0))
    pscale = {h: np.where(v > 0, v / 127.0, 1.0).astype(np.float32) for h, v in pmax.items()}
    Qf = {"fp16": Q, "int8": Q}
    for h, (mu, P, _) in fits.items():
        Qf[f"pca768-{h}"] = Qf[f"pca768-{h}+int8"] = proj(Q, mu, P)
    tops = {f: Top(Q.shape[0]) for f in FORMS}
    off_rows = 0
    for p in shards(ds):                                   # pass 3: every form from one read of each shard
        D = np.asarray(np.load(p, mmap_mode="r"), np.float32)
        tops["fp16"].add(Qf["fp16"], D, off_rows)
        tops["int8"].add(Qf["int8"], quant(D, scale8), off_rows)
        for h, (mu, P, _) in fits.items():
            Dp = proj(D, mu, P)
            tops[f"pca768-{h}"].add(Qf[f"pca768-{h}"], Dp, off_rows)
            tops[f"pca768-{h}+int8"].add(Qf[f"pca768-{h}+int8"], quant(Dp, pscale[h]), off_rows)
        off_rows += D.shape[0]
    hits = {}
    for f in FORMS:
        ti = tops[f].result()
        r5 = E.rrf_top(ti, spl, 5)
        hits[f] = {"d5": E.hits(ti[:, :5], flat, off), "d20": E.hits(ti[:, :20], flat, off),
                   "d1000": E.hits(ti, flat, off), "rrf5": E.hits(r5, flat, off)}
    # FS: fp16 through this search must equal the E0/E4 e3v read exactly
    z = np.load(E.OUT / "e0" / ENC / f"{ds}.npz")
    fs = {k: bool(np.array_equal(hits["fp16"][k], z[f"h_{k}"].astype(np.int64))) for k in ("d5", "d20", "d1000", "rrf5")}
    fs["ids"] = [str(x) for x in z["ids"]] == list(ids)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"read_{ds}.npz", ids=np.asarray(ids), gold_total=gt,
             **{f"{f}__{k}": v.astype(np.int16) for f in FORMS for k, v in hits[f].items()})
    gtf = gt.astype(np.float64)
    rec = {"stage": "ENC E5: read", "declared_in": "docs/ENC_ENCODER_COMPARISON.md section 21", "dataset": ds,
           "questions": len(ids), "fs": fs, "fs_call": "PASS" if all(fs.values()) else "FAIL",
           "variance_kept": {h: round(fits[h][2], 4) for h in fits},
           "recall": {f: {k: round(100 * float(np.mean(np.minimum(v, gt) / gtf)), 2) for k, v in hits[f].items()}
                      for f in FORMS},
           "npz_sha256": E.sha256(OUT / f"read_{ds}.npz"), "e0_read_sha256": E.sha256(E.OUT / "e0" / ENC / f"{ds}.npz"),
           "moments": {d: json.loads((OUT / f"moments_{d}.json").read_text(encoding="utf-8"))["npz_sha256"] for d in SIX},
           "code_sha256": code_sha(), "threads": E.THREADS, "seconds": round(time.time() - t0, 1), "utc": utc()}
    E.wjson(OUT / f"read_{ds}.json", rec)
    print(ds, rec["fs_call"], json.dumps(rec["recall"]), f"{rec['seconds']} s", flush=True)
    if rec["fs_call"] != "PASS":
        stop(f"FS failed on {ds}: {fs}")
    return 0


def paired(base, arm, boot=BOOT, seed=SEED):
    d = 100.0 * (arm - base)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, size=(boot, d.size))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return round(float(d.mean()), 2), round(float(lo), 2), round(float(hi), 2), (
        "ABOVE" if lo > 0 else ("BELOW" if hi < 0 else "LEVEL"))


def free(gaps):
    return all(g[3] in ("LEVEL", "ABOVE") and g[0] >= FREE_PTS for g in gaps)


def call(per):
    """21.4: the smallest form FREE on all six; at equal bytes loo before own (no fit on the corpus itself)."""
    order = sorted((f for f in FORMS if f != "fp16"), key=lambda f: (BYTES[f], "own" in f))
    for f in order:
        if all(per[ds][f]["free"] for ds in per):
            return f
    return "fp16"


def cmd_report(a):
    per, miss = {}, []
    for ds in SIX:
        jp = OUT / f"read_{ds}.json"
        if not jp.is_file():
            miss.append(ds)
            continue
        rec = json.loads(jp.read_text(encoding="utf-8"))
        if rec["fs_call"] != "PASS":
            stop(f"{ds}: FS did not pass")
        z = np.load(OUT / f"read_{ds}.npz")
        gt = z["gold_total"].astype(np.float64)
        r = lambda f, k: np.minimum(z[f"{f}__{k}"], z["gold_total"]) / gt  # noqa: E731
        per[ds] = {}
        for f in FORMS[1:]:
            g = {k: paired(r("fp16", k), r(f, k)) for k in KEYS}
            per[ds][f] = {"gaps": g, "free": free(g.values()), "recall": rec["recall"][f]}
        per[ds]["fp16"] = {"recall": rec["recall"]["fp16"]}
    if miss:
        stop(f"no read for {miss}")
    pick = call(per)
    rep = {"stage": "ENC E5: report", "declared_in": "docs/ENC_ENCODER_COMPARISON.md section 21",
           "rule": f"FREE on a dataset = dense R@5, dense R@1000 and RRF R@5 gaps to fp16 all LEVEL or ABOVE (paired 95% "
                   f"bootstrap, {BOOT}, seed {SEED}) with point gaps >= {FREE_PTS}; the call is the smallest form FREE "
                   f"on all six (loo before own at equal bytes), else fp16",
           "bytes_per_doc": BYTES, "call": pick, "per": per, "code_sha256": code_sha(), "utc": utc()}
    E.wjson(OUT / "report.json", rep)
    f = lambda g: f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] {g[3]}"  # noqa: E731
    L = ["# ENC E5: storage forms of e3v's document vectors (report)", "",
         "Written by `outputs/mp_unified/zenc5.py report` under docs/ENC_ENCODER_COMPARISON.md section 21. Gaps are the "
         "form minus fp16 in points (s1eval), paired 95% bootstrap over questions.", "",
         f"**Call: {pick}** ({BYTES[pick]:,} bytes a document, {BYTES[pick] / FROZEN_BYTES:.2f} of frozen's).", "",
         "| dataset | form | bytes | dense R@5 | R@5 gap | R@1000 gap | RRF R@5 gap | FREE |", "|---|---|---|---|---|---|---|---|"]
    for ds in SIX:
        L.append(f"| {ds} | fp16 | {BYTES['fp16']:,} | {per[ds]['fp16']['recall']['d5']:.2f} | | | | |")
        for fm in FORMS[1:]:
            x = per[ds][fm]
            L.append(f"| {ds} | {fm} | {BYTES[fm]:,} | {x['recall']['d5']:.2f} | {f(x['gaps']['d5'])} | "
                     f"{f(x['gaps']['d1000'])} | {f(x['gaps']['rrf5'])} | {'yes' if x['free'] else 'no'} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


def cmd_selftest(a):
    rng = np.random.default_rng(0)
    D = rng.standard_normal((3000, 64)).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    Q = D[:50] + 0.05 * rng.standard_normal((50, 64)).astype(np.float32)
    t = Top(50, k=100)
    for o in range(0, 3000, 700):
        t.add(Q, D[o:o + 700], o)
    S = Q @ D.T
    ref = np.lexsort((np.broadcast_to(np.arange(3000), S.shape), -S), axis=1)[:, :100]
    assert np.array_equal(t.result(), ref), "Top merge differs from a full sort"
    s = np.abs(D).max(0) / 127.0
    assert np.abs(quant(D, s) - D).max() <= s.max() / 2 + 1e-7
    g = paired(np.ones(200) * 0.5, np.ones(200) * 0.5)
    assert g[3] == "LEVEL" and free([g])
    per = {ds: {f: {"free": f in ("int8", "pca768-own+int8", "pca768-loo+int8")} for f in FORMS} for ds in SIX}
    assert call(per) == "pca768-loo+int8", call(per)
    per["metaqa"]["pca768-loo+int8"]["free"] = False
    assert call(per) == "pca768-own+int8", call(per)
    assert V.DIM == 1152 and BYTES["pca768-own+int8"] == 768
    print("zenc5 selftest: ok")
    if a.real:                       # the real shapes: SQuAD's fp16 through Top equals its E0 e3v read exactly
        ids, gt, flat, off, Q, spl = carve("squad")
        t, o = Top(Q.shape[0]), 0
        for p in shards("squad"):
            D = np.asarray(np.load(p, mmap_mode="r"), np.float32)
            t.add(Q, D, o)
            o += D.shape[0]
        ti = t.result()
        z = np.load(E.OUT / "e0" / ENC / "squad.npz")
        got = {"d5": E.hits(ti[:, :5], flat, off), "d1000": E.hits(ti, flat, off), "rrf5": E.hits(E.rrf_top(ti, spl, 5), flat, off)}
        bad = [k for k, v in got.items() if not np.array_equal(v, z[f"h_{k}"].astype(np.int64))]
        if bad or [str(x) for x in z["ids"]] != list(ids):
            stop(f"real selftest: squad differs from its E0 read on {bad}")
        print(f"zenc5 selftest --real: squad {len(ids):,} questions x {o:,} documents equal the E0 read", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc5.py")
    ap.add_argument("cmd", choices=["moments", "read", "report", "selftest"])
    ap.add_argument("--ds", choices=SIX)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    if a.cmd in ("moments", "read") and not a.ds:
        stop(f"{a.cmd} needs --ds")
    return {"moments": cmd_moments, "read": cmd_read, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
