"""P5a's starts and spaces made again, for the readers that need them (P5c5). At 00:08 on 19 October the host's clean-up
removed u_<dataset>.npy and u0_<dataset>.npy of the six corpora from outputs/p5a/space as builds (STAGE_BUILD's
"space", fixed in 82ebdb4). No archive held them.

P5a's own code (scripts/p5a_space.py, imported and never edited) runs again into outputs/p5a_restore, and each array
is checked against the sha256 that P5a's records name (outputs/p5a/space/base.json and space_<dataset>.json, committed).
Only an exact match goes back into outputs/p5a/space, and only where that file is absent. Nothing there is ever
overwritten, and P5a's records are not touched.

    python scripts/p5a_restore.py base [--host]
    python scripts/p5a_restore.py train DATASET [--host] [--device cuda] [--threads 2]
    python scripts/p5a_restore.py gate DATASET
    python scripts/p5a_restore.py install DATASET [DATASET ...]
    python scripts/p5a_restore.py --selftest

base: P5a's base stage as it is (the frame and the six starts) into outputs/p5a_restore/space; check_base.json.
train: P5a's train stage up to its space fit and no further, on the dataset's restored start, which must match. The
start's offsets, the evidence and the map are not made again: nothing reads them now. check_train_<dataset>.json.
gate: exit 0 only if the dataset's start and space both match their records.
install: the dataset's start and space, each only if it matches its record and only where absent; install_<dataset>.json.
It exits 3 unless both are then in place and match.

P5a ran under rx with 2 CPUs on 13 October: 2 BLAS threads (rx sets them from the CPUs) and --threads 2 on the card.
The items here ask the same, through the same wrappers. Whether the arrays come back bit for bit is what the checks
say. A start or space that does not is never installed, and P5c5 waits for it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import p5a_space as P  # noqa: E402  (sets the BLAS thread defaults as P5a did)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

REC = ROOT / "outputs" / "p5a" / "space"      # P5a's records (committed), and the arrays' home
RESTORE = ROOT / "outputs" / "p5a_restore"


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def redirect(out):
    """P5a's output paths, every one of them, moved under out: its stages write there and nowhere else."""
    out = Path(out)
    P.OUT = out
    P.SPACE = out / "space"
    P.FRAME = P.SPACE / "frame.npz"
    P.BASE = P.SPACE / "base.json"
    P.MAP = out / "map.npz"
    P.MAP_JSON = out / "map.json"
    P.READ1 = out / "read1"
    P.VERDICT = out / "verdict.json"
    P.VERDICT_MD = out / "verdict.md"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha_or_none(p):
    return P.sha_file(p) if Path(p).exists() else None


def same_code(want, what):
    got = P.script_sha256()
    if got != want["script_sha256"]:
        raise SystemExit(f"scripts/p5a_space.py is not the code that made {what} ({got[:12]}, the record "
                         f"{want['script_sha256'][:12]}): nothing is made again")


# ── base ─────────────────────────────────────────────────────────────────────


def check_base(rec_dir=REC, out=RESTORE):
    want = load(Path(rec_dir) / "base.json")
    sp = Path(out) / "space"
    g = sha_or_none(sp / "frame.npz")
    res = {"frame": {"sha256": g, "record": want["frame_sha256"], "matched": g == want["frame_sha256"]}, "u0": {}}
    for name, r in want["u0"].items():
        g = sha_or_none(sp / f"u0_{name}.npy")
        res["u0"][name] = {"sha256": g, "record": r["sha256"], "matched": g == r["sha256"]}
    return res


def base(host, rec_dir=REC, out=RESTORE):
    want = load(Path(rec_dir) / "base.json")
    same_code(want, "the starts")
    redirect(out)
    t0 = time.time()
    rc = P.base_stage(host)
    if rc:
        return rc
    made = load(P.BASE)
    res = check_base(rec_dir, out)
    rec = {"stage": "restore-base", "of": "outputs/p5a/space/base.json", **res,
           "freeze_RECORD_SHA256": made.get("freeze_RECORD_SHA256"),
           "freeze_matched": made.get("freeze_RECORD_SHA256") == want["freeze_RECORD_SHA256"],
           "matched": res["frame"]["matched"] and all(v["matched"] for v in res["u0"].values()),
           "p5a_script_sha256": P.script_sha256(), "script_sha256": script_sha256(),
           "seconds": round(time.time() - t0, 1), "peak_rss_bytes": P.peak_rss(), "utc": P.utc()}
    P.write_json(Path(out) / "check_base.json", rec)
    P.log("restore base: frame " + ("matches" if res["frame"]["matched"] else "DIFFERS") + "; starts "
          + ", ".join(f"{k} {'matches' if v['matched'] else 'DIFFERS'}" for k, v in res["u0"].items()))
    return 0


# ── train ────────────────────────────────────────────────────────────────────


def jsonable(x):
    return json.loads(json.dumps(x))


def train_record(name, info, u_path, rec_dir=REC, out=RESTORE, extra=None):
    """The made space against P5a's record: its sha256, and the fit's diagnostics beside it (they say how far a
    space that differs went apart)."""
    want = load(Path(rec_dir) / f"space_{name}.json")
    cb = load(Path(out) / "check_base.json")
    g = sha_or_none(u_path)
    rf = want["fit"]
    ours = jsonable({k: info.get(k) for k in ("gamma", "loss_hist", "delta_rms", "families")})
    rec = {"stage": "restore-train", "dataset": name, "of": f"outputs/p5a/space/space_{name}.json",
           "u_sha256": g, "record": want["u_sha256"], "matched": g == want["u_sha256"],
           "u0_matched": bool(cb["u0"][name]["matched"]),
           "fit_equal": {k: ours[k] == rf.get(k) for k in ours},
           "fit": {"gamma": ours["gamma"], "loss_hist": ours["loss_hist"], "delta_rms": ours["delta_rms"],
                   "seconds": info.get("seconds")},
           "record_fit": {"gamma": rf.get("gamma"), "loss_hist": rf.get("loss_hist"), "delta_rms": rf.get("delta_rms")},
           **(extra or {}), "script_sha256": script_sha256(), "utc": P.utc()}
    P.write_json(Path(out) / f"check_train_{name}.json", rec)
    return rec


def train(name, device, threads, host, rec_dir=REC, out=RESTORE):
    want = load(Path(rec_dir) / f"space_{name}.json")
    same_code(want, f"{name}'s space")
    cbp = Path(out) / "check_base.json"
    if not cbp.exists() or not load(cbp)["u0"][name]["matched"]:
        raise SystemExit(f"{name}: the restored start is missing or does not match P5a's: nothing is fit")
    redirect(out)
    import torch
    torch.set_num_threads(threads)                   # P5a's train stage, step for step, up to its space fit
    P.full_fp32(torch)
    P.modules()
    op = P.opened(host)
    if op.freeze["RECORD_SHA256"] != want["freeze_RECORD_SHA256"]:
        raise SystemExit(f"the substrate's record is {op.freeze['RECORD_SHA256'][:12]}, P5a's "
                         f"{want['freeze_RECORD_SHA256'][:12]}: nothing is fit")
    dev = torch.device(device)
    t0 = time.time()
    _ds, emb = P.dense_docs(op, name)
    u0p = P.SPACE / f"u0_{name}.npy"
    if P.sha_file(u0p) != want["u0_sha256"]:
        raise SystemExit(f"{name}: the restored start is not the one P5a's space was fit from")
    u0 = np.load(u0p)
    if u0.shape != (emb.n_rows, P.DIM):
        raise SystemExit(f"{name}: the start has {u0.shape}, the corpus {emb.n_rows} rows")
    fams = P.edge_families(name, P.csr_dir(op))
    if any(int(f.h.max(initial=-1)) >= u0.shape[0] or int(f.t.max(initial=-1)) >= u0.shape[0] for f in fams):
        raise SystemExit(f"{name}: an edge names a node outside the corpus")
    P.log(f"{name}: {u0.shape[0]} nodes, edges " + ", ".join(f"{f.name} {f.h.size}" for f in fams)
          + f" (restored start, {device})")
    u, _offs, info = P.fit_space(u0, fams, dev, steps=P.STEPS)
    u16 = u.half().cpu().numpy()
    del u
    up = P.SPACE / f"u_{name}.npy"
    P.save_npy(up, u16)
    rec = train_record(name, info, up, rec_dir, out, extra={
        "device": str(dev), "threads": threads, "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "gpu_peak_bytes": P.gpu_peak(torch, dev), "peak_rss_bytes": P.peak_rss(),
        "seconds": round(time.time() - t0, 1)})
    P.log(f"{name}: space " + ("MATCHES" if rec["matched"] else "DIFFERS from") + " P5a's record; the fit's "
          + ", ".join(f"{k} {'equal' if v else 'differ'}" for k, v in rec["fit_equal"].items()))
    return 0


# ── gate and install ─────────────────────────────────────────────────────────


def gate(name, out=RESTORE):
    p = Path(out) / f"check_train_{name}.json"
    if not p.exists():
        P.log(f"gate {name}: no check of the space")
        return 2
    r = load(p)
    ok = bool(r["matched"] and r["u0_matched"])
    P.log(f"gate {name}: start {'matches' if r['u0_matched'] else 'DIFFERS'}, space "
          f"{'matches' if r['matched'] else 'DIFFERS'}: " + ("the other spaces are made again" if ok else "STOP"))
    return 0 if ok else 3


def install(names, rec_dir=REC, out=RESTORE, dest=REC):
    base_rec = load(Path(rec_dir) / "base.json")
    worst = 0
    for name in names:
        want = {f"u0_{name}.npy": base_rec["u0"][name]["sha256"],
                f"u_{name}.npy": load(Path(rec_dir) / f"space_{name}.json")["u_sha256"]}
        res = {}
        for fn, w in want.items():
            src, dst = Path(out) / "space" / fn, Path(dest) / fn
            if dst.exists():
                g = P.sha_file(dst)
                res[fn] = "present, matches its record" if g == w else "present, differs from its record: left as it is"
            elif not src.exists():
                res[fn] = "not made"
            elif P.sha_file(src) != w:
                res[fn] = "made, differs from its record: not installed"
            else:
                os.replace(src, dst)
                res[fn] = "installed" if P.sha_file(dst) == w else "installed, then differs: CHECK"
        ok = all(v in ("installed", "present, matches its record") for v in res.values())
        P.write_json(Path(out) / f"install_{name}.json",
                     {"stage": "restore-install", "dataset": name, "files": res, "in_place": ok,
                      "script_sha256": script_sha256(), "utc": P.utc()})
        P.log(f"install {name}: " + "; ".join(f"{k} {v}" for k, v in res.items()))
        worst = max(worst, 0 if ok else 3)
    return worst


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    import shutil
    import tempfile
    import torch
    saved = {k: getattr(P, k) for k in ("OUT", "SPACE", "FRAME", "BASE", "MAP", "MAP_JSON", "READ1", "VERDICT",
                                        "VERDICT_MD")}
    tmp = Path(tempfile.mkdtemp(prefix="p5ar_"))
    try:
        # 1. the frozen script is the one P5a's records name
        for f in ["base.json"] + [f"space_{d}.json" for d in P.SIX]:
            if (REC / f).exists():
                assert load(REC / f)["script_sha256"] == P.script_sha256(), f
        # 2. redirect moves every output path P5a writes, and nothing it reads
        redirect(tmp / "r")
        assert P.SPACE == tmp / "r" / "space" and P.FRAME.parent == P.SPACE and P.BASE.parent == P.SPACE
        assert all(Path(getattr(P, k)).is_relative_to(tmp) for k in saved), "an output path stayed in place"
        assert P.CSR == ROOT / "outputs" / "m3b" / "csr" and P.load_frame() is None
        mu, Pm = np.zeros(8, np.float32), np.eye(64, 8, dtype=np.float32).T
        P.save_npz(P.FRAME, mu=mu, P=Pm, eigenvalues=np.ones(8))
        assert np.array_equal(P.load_frame()[1], Pm)
        # 3. P5a's fit, through this module, twice on the CPU: the same bytes
        rng = np.random.default_rng(1)
        n = 300
        u0 = rng.standard_normal((n, P.DIM)).astype(np.float16)
        def fams():
            r = np.random.default_rng(2)
            return [P.Family("structural", r.integers(0, n, 900), r.integers(0, n, 900), signed=False),
                    P.Family("ner", r.integers(0, n, 600), r.integers(0, n, 600)),
                    P.Family("knn", r.integers(0, n, 600), r.integers(0, n, 600))]
        torch.set_num_threads(1)
        outs = []
        for _ in range(2):
            u, _o, info = P.fit_space(u0, fams(), torch.device("cpu"), steps=4, batch=384, logf=lambda m: None)
            outs.append((u.half().numpy().tobytes(), info))
        assert outs[0][0] == outs[1][0] and outs[0][1]["loss_hist"] == outs[1][1]["loss_hist"]
        u16 = np.frombuffer(outs[0][0], np.float16).reshape(n, P.DIM)
        # 4. the checks, the gate and the install on a toy pair of datasets
        rec_dir, out, dest = tmp / "rec", tmp / "restore", tmp / "dest"
        for d in (rec_dir, out / "space", dest):
            d.mkdir(parents=True)
        redirect(out)
        P.save_npz(P.FRAME, mu=mu, P=Pm, eigenvalues=np.ones(8))
        P.save_npy(out / "space" / "u0_d1.npy", u0)
        P.save_npy(out / "space" / "u0_d2.npy", u0[::-1].copy())
        P.save_npy(out / "space" / "u_d1.npy", u16)
        good_u0 = P.sha_file(out / "space" / "u0_d1.npy")
        good_u = P.sha_file(out / "space" / "u_d1.npy")
        code = {"script_sha256": P.script_sha256(), "freeze_RECORD_SHA256": "x"}
        P.write_json(rec_dir / "base.json", {**code, "frame_sha256": P.sha_file(P.FRAME),
                                             "u0": {"d1": {"sha256": good_u0}, "d2": {"sha256": good_u0}}})
        for d in ("d1", "d2"):
            P.write_json(rec_dir / f"space_{d}.json", {**code, "u_sha256": good_u, "u0_sha256": good_u0,
                                                       "fit": jsonable(outs[0][1])})
        cb = check_base(rec_dir, out)
        assert cb["frame"]["matched"] and cb["u0"]["d1"]["matched"] and not cb["u0"]["d2"]["matched"]
        P.write_json(out / "check_base.json", cb)
        r1 = train_record("d1", outs[0][1], out / "space" / "u_d1.npy", rec_dir, out)
        assert r1["matched"] and r1["u0_matched"] and all(r1["fit_equal"].values()), r1["fit_equal"]
        assert gate("d1", out) == 0 and gate("d2", out) == 2
        P.save_npy(out / "space" / "u_d2.npy", u16[::-1].copy())
        r2 = train_record("d2", outs[0][1], out / "space" / "u_d2.npy", rec_dir, out)
        assert not r2["matched"] and gate("d2", out) == 3
        try:
            train("d2", "cpu", 1, False, rec_dir, out)
            raise AssertionError("a space was fit on a start that does not match")
        except SystemExit as e:
            assert "does not match" in str(e)
        # install: matches go in, nothing is overwritten, a mismatch stays out
        assert install(["d1"], rec_dir, out, dest) == 0
        assert P.sha_file(dest / "u_d1.npy") == good_u and not (out / "space" / "u_d1.npy").exists()
        assert install(["d1"], rec_dir, out, dest) == 0          # present and matching: left, and in place
        P.save_npy(dest / "u0_d2.npy", u0)                         # present (here, it matches its record)
        before = P.sha_file(dest / "u0_d2.npy")
        assert install(["d2"], rec_dir, out, dest) == 3
        st = load(out / "install_d2.json")["files"]
        assert st["u0_d2.npy"] == "present, matches its record"
        assert st["u_d2.npy"] == "made, differs from its record: not installed"
        assert (out / "space" / "u_d2.npy").exists() and not (dest / "u_d2.npy").exists()
        P.save_npy(dest / "u_d2.npy", u16[::-1].copy())          # present and different: never overwritten
        assert install(["d2"], rec_dir, out, dest) == 3
        assert load(out / "install_d2.json")["files"]["u_d2.npy"].startswith("present, differs")
        assert P.sha_file(dest / "u0_d2.npy") == before
        # 5. a changed P5a script refuses to make anything
        P.write_json(rec_dir / "base.json", {**load(rec_dir / "base.json"), "script_sha256": "0" * 64})
        try:
            base(False, rec_dir, out)
            raise AssertionError("the base stage ran under another script")
        except SystemExit as e:
            assert "is not the code" in str(e)
    finally:
        for k, v in saved.items():
            setattr(P, k, v)
        shutil.rmtree(tmp, ignore_errors=True)
    P.log("p5a_restore selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("base", "train", "gate", "install"))
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if any(d not in P.SIX for d in a.datasets):
        raise SystemExit(f"datasets must be among {P.SIX}")
    if a.stage == "base":
        return base(a.host)
    if a.stage in ("train", "gate") and len(a.datasets) != 1:
        raise SystemExit(f"{a.stage} takes one dataset")
    if a.stage == "train":
        return train(a.datasets[0], a.device, a.threads, a.host)
    if a.stage == "gate":
        return gate(a.datasets[0])
    if a.stage == "install":
        return install(a.datasets or list(P.SIX))
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
