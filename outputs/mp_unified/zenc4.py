"""ENC E4 (docs/ENC_ENCODER_COMPARISON.md section 20): e3v's vectors on the four other datasets (hotpotqa, 2wiki, webqsp,
metaqa), after D101 (the user took e3v, 21 Nov 2026). E0N's encode (zenc3.py), zenc3v's fusion and zenc's first stage
run unchanged; this module only widens their dataset lists in its own process (their files are not edited).

    python outputs/mp_unified/zenc4.py encode --enc gtemodern|granitesm --ds DS [--threads N]   GPU: one part, one dataset
    python outputs/mp_unified/zenc4.py vectors --ds DS [--threads N]       fuse the two parts' shards into e3v's
    python outputs/mp_unified/zenc4.py fc --ds DS [--threads N]            FC: e3v's shards against the parts, exactly
    python outputs/mp_unified/zenc4.py release-parts --ds DS [--apply]     the parts' shards go once FC has passed
    python outputs/mp_unified/zenc4.py first --enc e3v|frozen --ds DS [--threads N]
    python outputs/mp_unified/zenc4.py report [--threads N]                the six datasets' first stage, e3v beside frozen
    python outputs/mp_unified/zenc4.py selftest
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import zenc as E  # noqa: E402  (reads --threads from sys.argv before numpy loads)
import zenc3 as Z3  # noqa: E402  (registers gtemodern, granite, granitesm)
import zenc3v as V  # noqa: E402  (registers e3v)
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

NEW4 = ("hotpotqa", "2wiki", "webqsp", "metaqa")
ALL6 = ("squad", "musique") + NEW4
OUT = E.OUT / "e4"
BOOT, SEED = 2000, 0
ROOTWS = E.OUT.parents[1]


def rel(p):
    return Path(p).resolve().relative_to(ROOTWS).as_posix()


def stop(msg):
    E.stop(f"E4: {msg}")


def need_new(ds):
    if ds not in NEW4:
        stop(f"{ds} is not one of E4's datasets {NEW4}")


def cmd_encode(a):
    need_new(a.ds)
    if a.enc not in V.PARTS:
        stop(f"--enc must be one of e3v's parts {V.PARTS}")
    Z3.DS = (a.ds,)
    return Z3.cmd_encode(argparse.Namespace(enc=a.enc))


def cmd_vectors(a):
    need_new(a.ds)
    V.DS = ALL6
    return V.main(["vectors", "--ds", a.ds, "--threads", str(a.threads)])


def fc_ds(ds, rows_per_shard=256):
    """FC (20.3): every e3v shard hashes to its manifest; every shard's recorded sources hash to the parts' files now;
    and the first rows of every shard equal zenc3v.fuse of the parts' rows exactly."""
    root = E.served_root()
    out = {"dataset": ds, "kinds": {}}
    ok = True
    for kind in ("docs", "queries"):
        man = json.loads((E.OUT / V.ENC / ds / kind / "manifest.json").read_text(encoding="utf-8"))
        srcs = {e: E.vec_shards(root, e, ds, kind) for e in V.PARTS}
        k = {"shards": len(man["shards"]), "rows": int(man["n_rows"]), "bad_sha": [], "bad_source": [], "bad_rows": []}
        for s, sh in enumerate(man["shards"]):
            npy = E.OUT / V.ENC / ds / kind / sh["file"]
            if E.sha256(npy) != sh["sha256"]:
                k["bad_sha"].append(s)
            rec = json.loads(npy.with_suffix(".json").read_text(encoding="utf-8"))
            for e in V.PARTS:
                if E.sha256(srcs[e][s]) != rec["sources"][e]["sha256"]:
                    k["bad_source"].append([s, e])
            X = np.load(npy, mmap_mode="r")
            n = min(rows_per_shard, int(X.shape[0]))
            want = V.fuse([np.load(srcs[e][s], mmap_mode="r")[:n] for e in V.PARTS])
            if not np.array_equal(np.asarray(X[:n]), want):
                k["bad_rows"].append(s)
        k["verdict"] = "PASS" if not (k["bad_sha"] or k["bad_source"] or k["bad_rows"]) else "FAIL"
        ok &= k["verdict"] == "PASS"
        out["kinds"][kind] = k
    out["verdict"] = "PASS" if ok else "FAIL"
    return out


def cmd_fc(a):
    need_new(a.ds)
    t0 = time.time()
    r = fc_ds(a.ds)
    r.update({"stage": "ENC E4: FC", "declared_in": "docs/ENC_ENCODER_COMPARISON.md section 20.3",
              "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    OUT.mkdir(parents=True, exist_ok=True)
    E.wjson(OUT / f"fc_{a.ds}.json", r)
    print(f"FC {a.ds}: {r['verdict']} " + json.dumps({k: {x: v[x] for x in ('shards', 'rows', 'verdict')}
                                                       for k, v in r["kinds"].items()}), flush=True)
    return 0 if r["verdict"] == "PASS" else 3


def cmd_release_parts(a):
    """20.4: the parts' shard arrays of one E4 dataset go once its FC has passed and the e3v shards still hash to the
    manifest. The parts' shard json and manifests stay (they name every removed file and its sha256)."""
    need_new(a.ds)
    fp = OUT / f"fc_{a.ds}.json"
    if not fp.is_file() or json.loads(fp.read_text(encoding="utf-8")).get("verdict") != "PASS":
        stop(f"FC has not passed for {a.ds}")
    root = E.served_root()
    for kind in ("docs", "queries"):
        man = json.loads((E.OUT / V.ENC / ds_kind(a.ds, kind) / "manifest.json").read_text(encoding="utf-8"))
        for sh in man["shards"]:
            if E.sha256(E.OUT / V.ENC / a.ds / kind / sh["file"]) != sh["sha256"]:
                stop(f"e3v {a.ds} {kind} {sh['file']} no longer hashes to its manifest")
    items = []
    for e in V.PARTS:
        for kind in ("docs", "queries"):
            for p in E.vec_shards(root, e, a.ds, kind):
                p = Path(p)
                if E.OUT not in p.resolve().parents:
                    stop(f"{p} is outside outputs/enc")
                j = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
                items.append({"path": rel(p), "bytes": p.stat().st_size, "sha256": j["sha256"]})
    gb = sum(i["bytes"] for i in items) / 2 ** 30
    print(f"release-parts {a.ds}: {len(items)} part shards, {gb:.2f} GB", flush=True)
    if not a.apply:
        print("dry run: nothing removed")
        return 0
    import shutil
    free0 = shutil.disk_usage(E.OUT).free / 2 ** 30
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rp = ROOTWS / "outputs" / "host_cleanup" / f"deleted_e4parts_{a.ds}_{stamp}.json"
    rec = {"stage": "ENC E4: release-parts", "declared_in": "docs/ENC_ENCODER_COMPARISON.md section 20.4",
           "dataset": a.ds, "fc": rel(fp), "free_before_gb": round(free0, 2), "items": items}
    E.wjson(rp, rec)
    for i in items:
        os.unlink(ROOTWS / i["path"])
    rec["free_after_gb"] = round(shutil.disk_usage(E.OUT).free / 2 ** 30, 2)
    E.wjson(rp, rec)
    print(f"removed {len(items)}; free {rec['free_before_gb']} -> {rec['free_after_gb']} GB; {rp.name}", flush=True)
    return 0


def ds_kind(ds, kind):
    return f"{ds}/{kind}"


def cmd_first(a):
    if a.ds not in ALL6:
        stop(f"{a.ds} is not one of the six")
    if a.enc == "e3v":
        V.DS = ALL6
        return V.main(["first", "--ds", a.ds, "--threads", str(a.threads)])
    if a.enc == "frozen":
        return E.cmd_first(argparse.Namespace(enc="frozen", ds=a.ds)) or 0
    stop("--enc must be e3v or frozen")


def paired(rf, ra, boot=BOOT, seed=SEED):
    """Point gap (arm - frozen, points) and its paired 95% bootstrap interval over questions."""
    d = 100.0 * (ra - rf)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(boot, d.size))
    m = d[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    lab = "ABOVE" if lo > 0 else ("BELOW" if hi < 0 else "LEVEL")
    return round(float(d.mean()), 2), round(float(lo), 2), round(float(hi), 2), lab


def read(enc, ds):
    z = np.load(E.OUT / "e0" / enc / f"{ds}.npz")
    gt = z["gold_total"].astype(np.float64)
    r = {k: np.minimum(z[f"h_{k}"], z["gold_total"]) / gt for k in ("d5", "d20", "d1000")}
    return list(z["ids"]), r, E.sha256(E.OUT / "e0" / enc / f"{ds}.npz")


def cmd_report(a):
    rows, miss = [], []
    for ds in ALL6:
        try:
            idf, rf, sf = read("frozen", ds)
            ida, ra, sa = read("e3v", ds)
        except FileNotFoundError as ex:
            miss.append(f"{ds}: {ex.filename}")
            continue
        if idf != ida:
            stop(f"{ds}: the two reads have different questions")
        row = {"dataset": ds, "questions": len(idf), "npz": {"frozen": sf, "e3v": sa}}
        for k in ("d5", "d20", "d1000"):
            row[k] = {"frozen": round(100 * float(rf[k].mean()), 2), "e3v": round(100 * float(ra[k].mean()), 2),
                      "gap": paired(rf[k], ra[k])}
        rows.append(row)
    rep = {"stage": "ENC E4: report", "declared_in": "docs/ENC_ENCODER_COMPARISON.md section 20.5",
           "rule": "beside, no call: s1eval dense R@k x100, e3v minus frozen, paired 95% bootstrap (2,000, seed 0)",
           "rows": rows, "missing": miss, "fc": {}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for ds in NEW4:
        fp = OUT / f"fc_{ds}.json"
        rep["fc"][ds] = json.loads(fp.read_text(encoding="utf-8"))["verdict"] if fp.is_file() else None
    OUT.mkdir(parents=True, exist_ok=True)
    E.wjson(OUT / "report.json", rep)
    f = lambda g: f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] {g[3]}"  # noqa: E731
    L = ["# ENC E4: e3v's first stage on the six datasets, beside frozen (report)", "",
         "Written by `outputs/mp_unified/zenc4.py report` under docs/ENC_ENCODER_COMPARISON.md section 20. s1eval dense "
         "R@k x100; a gap is e3v minus frozen in points, paired 95% bootstrap over questions. Beside, no call.", "",
         "| dataset | questions | frozen R@5 | e3v R@5 | R@5 gap | R@20 gap | R@1000 gap |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['questions']:,} | {r['d5']['frozen']:.2f} | {r['d5']['e3v']:.2f} | "
                 f"{f(r['d5']['gap'])} | {f(r['d20']['gap'])} | {f(r['d1000']['gap'])} |")
    L += ["", "FC: " + ", ".join(f"{k} {v}" for k, v in rep["fc"].items())] + (["", "Missing: " + "; ".join(miss)] if miss else [])
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0 if not miss else 3


def cmd_selftest(a):
    assert set(NEW4) | {"squad", "musique"} == set(E.SIX), E.SIX
    assert V.PARTS == ("gtemodern", "granitesm") or set(V.PARTS) == {"gtemodern", "granitesm"}, V.PARTS
    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal((8, E.ENCODERS[e]["dim"])).astype(np.float16) for e in V.PARTS]
    X = V.fuse(blocks)
    assert X.shape == (8, V.DIM) and X.dtype == np.float16
    rf = rng.random(500)
    g = paired(rf, rf)
    assert g[0] == 0.0 and g[3] == "LEVEL", g
    g = paired(rf, np.minimum(rf + 0.05, 1.0))
    assert g[3] == "ABOVE" and g[1] > 0, g
    V.DS = ALL6
    Z3.DS = ("metaqa",)
    print("zenc4 selftest: ok")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc4.py")
    ap.add_argument("cmd", choices=["encode", "vectors", "fc", "release-parts", "first", "report", "selftest"])
    ap.add_argument("--enc")
    ap.add_argument("--ds")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd in ("encode", "vectors", "fc", "release-parts", "first") and not a.ds:
        stop(f"{a.cmd} needs --ds")
    return {"encode": cmd_encode, "vectors": cmd_vectors, "fc": cmd_fc, "release-parts": cmd_release_parts,
            "first": cmd_first, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
