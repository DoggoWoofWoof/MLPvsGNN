"""FSV's timing (docs/FSV_FAST_VERSIONS.md, section 5, with its timing amendment): both families of Claim 3 timed cold
and warm on the same questions, each version at its fastest exact form.

  family 1   the full models: zrckpn (LTM's MLP of record) and zg6r (ML1's GNN), every input kept
  family 2   FSV's fast versions: each model's first ADMITTED level, or the full model where none is admitted

This file holds the reference read first (the code of the timed path and its development check come before any timed
number, in the amendment):

  ref   one fit's p@swa scores on one dataset's first 201 s1eval questions (the 200 timed ones and the JIT question), on
        the CPU, read through the fit's own read chain (zltm's for zrckpn, zml1s's for zg6r) with lean_gpu.read_cmd
        replaced for the call by a read that writes only under outputs/fsv/timing/ref: the same carve class, model class,
        pick, roots and read cap as the fit's read, read_carve's batches, then every question again at batch 1. Checked
        against the fit's filed read file (its scores64 on the first 64 questions, its top and hit on all 201). Nothing
        under outputs/screen/fits is written, and no stamp is touched.

The fits each dataset is timed on are the in-domain ones of the six-read rule (6cb8f39): musique on WQ-hotpotqa's fits,
hotpotqa and 2wiki on WQ-musique's.

    python outputs/mp_unified/zfsvt.py ref --model mlp|gnn --dataset D          (the host)
"""
import os
import sys

THREADS_REF = 6                                  # zrckpn's read (LTM's CPU device); the GNN's reference uses the same
if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = str(THREADS_REF)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

DECLARED = "docs/FSV_FAST_VERSIONS.md"
STAGE = "FSV"
OUT = ROOT / "outputs" / "fsv" / "timing"
CARVE = "s1eval"
NQ = 200                                         # timed questions per dataset (section 5)
NREF = NQ + 1                                    # and the JIT question, the carve's 201st
CAND = "p@swa"
DATASETS = ("musique", "hotpotqa", "2wiki")
MODELS = ("mlp", "gnn")
ARM = {"mlp": "zrckpn", "gnn": "zg6r"}
FITS = {("mlp", "musique"): "scr-ltm-zrckpn-s0-hp", ("mlp", "hotpotqa"): "scr-ltm-zrckpn-s0",
        ("mlp", "2wiki"): "scr-ltm-zrckpn-s0",
        ("gnn", "musique"): "scr-ml1-zg6r-s0-hp", ("gnn", "hotpotqa"): "scr-ml1-zg6r-s0",
        ("gnn", "2wiki"): "scr-ml1-zg6r-s0"}
TOL = 1e-4                                       # C3's exactness: scores within 1e-4 of the filed read, the same top 5
assert set(FITS) == {(m, d) for m in MODELS for d in DATASETS}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsvt: {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def ref_dir(fit):
    return OUT / "ref" / fit


# ── the reference read ───────────────────────────────────────────────────────


def top5(s):
    """A question's five highest rows, lean_gpu.top_hit's order (descending score, NaN last, ties by pool position)."""
    import numpy as np
    s = np.asarray(s, np.float32)
    key = np.where(np.isnan(s), np.inf, -s.astype(np.float64)) + 0.0
    return np.argsort(key, kind="stable")[:5]


def read_ref(LG, a, ds, nq):
    """In place of lean_gpu.read_cmd, under the fit's read chain: the fit's p@swa model and carve on the CPU, read_carve's
    batches over the questions holding the first nq, then each of them at batch 1. Returns the arrays and the record;
    writes nothing."""
    import numpy as np
    import torch
    t0 = time.time()
    flags = LG.set_flags("cpu")
    LG.bind_device_ops()
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()
    fdir = Path(a.out_root or LG.OUT) / a.name
    blob = torch.load(fdir / "models.pt", weights_only=False)
    names = [str(c) for c in blob["candidates"]]
    if CAND not in names:
        raise SystemExit(f"zfsvt: {a.name} holds no {CAND} ({names})")
    blob["candidates"] = [CAND]
    models = LG.load_models(blob, "cpu")
    if len(models) != 1 or models[0][0] != CAND:
        raise SystemExit(f"zfsvt: load_models built {[m[0] for m in models]}, not {CAND}")
    _nm, m, bl = models[0]
    cache_root = Path(a.cache_root) if a.cache_root else LG.LC.OUT
    c = LG.CacheCarve(ds, CARVE, blob["basis"], "cpu", cache_root, True, score2=True)
    if c.basis_sha256 != blob["basis_sha256"]:
        raise SystemExit(f"zfsvt: {ds}/{CARVE}: the cache's basis is not the fit's")
    blocks = [b for b in LG.SETS["pick"] if b in bl]
    if blocks != list(bl):
        raise SystemExit(f"zfsvt: the pick {LG.SETS['pick']} orders the model's blocks {list(bl)} as {blocks}")
    need = min(nq, int(c.rows))
    off = np.asarray(c.off_np, np.int64)
    n = np.asarray(c.n_np[:need], np.int64)
    scores = np.zeros(int(off[need]), np.float32)
    top = np.zeros(need, np.int16)
    hit = np.zeros(need, np.uint8)
    inventory = None
    with torch.no_grad():
        for q0, q1 in LG.batches(c.n_np, a.rows_cap):
            if q0 >= need:
                break
            qs = np.arange(q0, q1)
            B = q1 - q0
            feats, nqv, base_z, gold = c.batch(qs, blocks)
            if inventory is None:
                inventory = {k: ([list(x.shape[1:]) for x in v] if isinstance(v, tuple) else list(v.shape[1:]))
                             if isinstance(v, (tuple, torch.Tensor)) else type(v).__name__ for k, v in feats.items()}
            cnt = torch.from_numpy(c.n_np[q0:q1])
            seg = torch.cumsum(cnt, 0) - cnt
            s = m(feats, torch.ones((B, len(bl)), dtype=torch.float32), nqv, B, base_z)
            t, h = LG.top_hit(s, gold, B, seg, cnt)
            e = min(q1, need)
            r0, r1 = int(off[q0]), int(off[e])
            scores[r0:r1] = s[:r1 - r0].numpy()
            top[q0:e] = t[:e - q0].numpy()
            hit[q0:e] = h[:e - q0].numpy()
        b1 = np.zeros_like(scores)
        for q in range(need):
            feats, nqv, base_z, _gold = c.batch(np.asarray([q]), blocks)
            s = m(feats, torch.ones((1, len(bl)), dtype=torch.float32), nqv, 1, base_z)
            b1[int(off[q]):int(off[q + 1])] = s.numpy()
    ids = [str(x) for x in list(c.ids)[:need]]
    rec = {"fit": a.name, "dataset": ds, "carve": CARVE, "candidate": CAND, "candidates_in_blob": names,
           "models_sha256": sha_file(fdir / "models.pt"), "basis": blob["basis"], "basis_sha256": blob["basis_sha256"],
           "carve_class": type(c).__name__, "model_class": type(m).__name__, "blocks": list(bl),
           "widths": {k: int(v) for k, v in dict(c.widths).items()}, "feats": inventory, "rows_cap": int(a.rows_cap),
           "cache_root": str(a.cache_root), "parts": dict(getattr(c, "part_shas", {}) or {}),
           "carve_ids_sha256": (getattr(c, "look", None) or {}).get("carve_ids_sha256"), "carve_questions": int(c.rows),
           "questions": need, "rows": int(off[need]), "flags": flags, "placement": placement,
           "torch_threads": torch.get_num_threads(), "seconds": round(time.time() - t0, 1)}
    return {"ids": ids, "n": n, "scores": scores, "scores_b1": b1, "top": top, "hit": hit}, rec


@contextlib.contextmanager
def capturing(LG, ds, nq, got):
    """lean_gpu.read_cmd replaced by read_ref for the block; put back after, also on a failure."""
    orig = LG.read_cmd

    def read_cmd(a):
        got.append(read_ref(LG, a, ds, nq))
        return 0

    LG.read_cmd = read_cmd
    try:
        yield
    finally:
        LG.read_cmd = orig


def ref_mlp(fit, ds, nq, got):
    """zltm.read's chain for an LTM fit (on_ltm, on_ml1, on_wq, the arm swap and the split's NEXT columns), without its
    stamp: zp1c3p's read reaches read_ref."""
    import zltm as LTM
    M1 = LTM.M1
    LTM.import_arm()
    with LTM.on_ltm():
        if fit not in LTM.FITS:
            raise SystemExit(f"zfsvt: {fit} is not one of LTM's fits")
        _arm, _seed, split = M1.ALL_FITS[fit]
        _rec, k = M1.fit_stamp(fit, what="read")
        nk = k.get("next") or {}
        if (nk.get("split"), nk.get("columns")) != (split, list(LTM.LTMC.COLS)) or not nk.get("fit_sha256"):
            raise SystemExit(f"zfsvt: {fit}'s read stamp names no NEXT columns of {split} ({nk})")
        argv = ["--name", fit, "--device", "cpu", "--host", "--cache-root", M1.ROOTS["cache"],
                "--rows-cap", str(LTM.ROWS_CAP)]
        M1._ACTIVE[0] = fit
        LTM.LOADED.clear()
        try:
            with M1.on_ml1() as WQ:
                WQ.cache_root_ok(M1.ROOTS["cache"])
                WQ.gate_passed()
                why = WQ.wq_why(fit)
                if why:
                    raise SystemExit(f"zfsvt: {why}")
                with WQ.on_wq() as W, LTM.on_arm(W), LTM.on_split(split, nk["fit_sha256"]), \
                        capturing(W.LG, ds, nq, got):
                    rc = W.Z3P.read(argv)
            loaded = [dict(x) for x in LTM.LOADED]
        finally:
            M1._ACTIVE[0] = None
            LTM.LOADED.clear()
    return rc, {"split": split, "next": nk, "next_loaded": loaded}


def ref_gnn(fit, ds, nq, got):
    """zml1s.run_arm's read for an ML1 zg6r fit (on_ml1, on_wq, the typed links, the arm's checks), without zml1s.read's
    stamp: zrm's read reaches read_ref."""
    import zml1s as M1
    import lean_gpu as LG
    if fit not in M1.ALL_FITS or M1.ALL_FITS[fit][0] != "zg6r":
        raise SystemExit(f"zfsvt: {fit} is not one of ML1's zg6r fits")
    _arm, _seed, split = M1.ALL_FITS[fit]
    M1.fit_stamp(fit, what="read")
    with capturing(LG, ds, nq, got):
        rc, extra = M1.run_arm(fit, None, "read")
    return rc, {"split": split, "records": {k: v for k, v in extra.items() if k != "encoder"}}


def ref_check(arrs, rec, fdir, ds):
    """The reference against the fit's filed read file: its ids, scores64 on the first 64 questions, top and hit on every
    question read; and batch 1 against read_carve's batches."""
    import numpy as np
    p = Path(fdir) / "reads" / f"{ds}__{CARVE}.npz"
    z = np.load(p)
    cands = [str(x) for x in z["candidates"]]
    k = cands.index(CAND)
    need = rec["questions"]
    n64, rows64 = int(z["n64"]), int(z["rows64"])
    ids_ok = [str(x) for x in z["ids"][:need]] == arrs["ids"]
    m64 = min(n64, need)
    r64 = int(np.sum(arrs["n"][:m64]))
    if m64 == n64 and r64 != rows64:
        raise SystemExit(f"zfsvt: the filed read's first {n64} questions hold {rows64} rows, the carve's {r64}")
    d64 = np.abs(arrs["scores"][:r64].astype(np.float64) - z["scores64"][k, :r64].astype(np.float64))
    db1 = np.abs(arrs["scores_b1"].astype(np.float64) - arrs["scores"].astype(np.float64))
    off = np.r_[0, np.cumsum(arrs["n"])]
    t5_64 = sum(np.array_equal(top5(arrs["scores"][off[q]:off[q + 1]]), top5(z["scores64"][k, off[q]:off[q + 1]]))
                for q in range(m64))
    t5_b1 = sum(np.array_equal(top5(arrs["scores"][off[q]:off[q + 1]]), top5(arrs["scores_b1"][off[q]:off[q + 1]]))
                for q in range(need))
    return {"read_file": str(p.relative_to(ROOT)).replace("\\", "/"), "read_file_sha256": sha_file(p), "ids_equal": ids_ok,
            "scores64_questions": m64, "scores64_max_abs": float(d64.max()) if d64.size else 0.0,
            "scores64_exact": bool(d64.size and float(d64.max()) == 0.0), "scores64_top5_same": int(t5_64),
            "top_equal": int((z["top"][k, :need] == arrs["top"]).sum()),
            "hit_equal": int((z["hit"][k, :need] == arrs["hit"]).sum()),
            "batch1_max_abs": float(db1.max()) if db1.size else 0.0, "batch1_top5_same": int(t5_b1),
            "pass": bool(ids_ok and (not d64.size or float(d64.max()) <= TOL) and t5_64 == m64
                         and int((z["top"][k, :need] == arrs["top"]).sum()) == need
                         and int((z["hit"][k, :need] == arrs["hit"]).sum()) == need
                         and float(db1.max()) <= TOL and t5_b1 == need)}


def ref_cmd(model, ds, nq=NREF):
    import numpy as np
    import torch
    torch.set_num_threads(THREADS_REF)
    fit = FITS[(model, ds)]
    got = []
    t0 = time.time()
    rc, ctx = (ref_mlp if model == "mlp" else ref_gnn)(fit, ds, nq, got)
    if rc != 0 or len(got) != 1:
        raise SystemExit(f"zfsvt: the read chain of {fit} exited {rc} with {len(got)} captured reads")
    arrs, rec = got[0]
    fdir = ROOT / "outputs" / "screen" / "fits" / fit
    chk = ref_check(arrs, rec, fdir, ds)
    out = ref_dir(fit)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"{ds}.npz"
    tmp = out / f"{ds}.tmp.npz"
    np.savez_compressed(tmp, ids=np.asarray(arrs["ids"]), n=arrs["n"], scores=arrs["scores"],
                        scores_b1=arrs["scores_b1"], top=arrs["top"], hit=arrs["hit"])
    os.replace(tmp, npz)
    full = {"declared_in": DECLARED, "stage": STAGE, "what": "reference read", "model": model, "arm": ARM[model],
            **rec, "chain": ctx, "check": chk, "npz": npz.name, "npz_sha256": sha_file(npz),
            "script_sha256": sha_src(__file__), "seconds_total": round(time.time() - t0, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / f"{ds}.json", full)
    log(f"ref {fit} {ds}: {rec['questions']} questions, {rec['rows']} rows; filed scores64 max |d| "
        f"{chk['scores64_max_abs']:.3g} (top5 {chk['scores64_top5_same']}/{chk['scores64_questions']}), top "
        f"{chk['top_equal']}, hit {chk['hit_equal']}; batch 1 max |d| {chk['batch1_max_abs']:.3g} "
        f"(top5 {chk['batch1_top5_same']}); {'PASS' if chk['pass'] else 'FAIL'} ({full['seconds_total']:.0f}s)")
    log(f"  blocks {rec['blocks']}; feats {rec['feats']}")
    return 0 if chk["pass"] else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ref",))
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--questions", type=int, default=NREF)
    a = ap.parse_args(argv)
    if a.cmd == "ref":
        return ref_cmd(a.model, a.dataset, a.questions)
    return 1


if __name__ == "__main__":
    sys.exit(main())
