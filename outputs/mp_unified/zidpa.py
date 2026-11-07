"""IDP amendment A1 (docs/IDP_IDENTICAL_INPUT_PAIR.md section 5): GNX, the GNN that reads the MLP of record's inputs, read on
B1P's six carves (HippoRAG 2's musique, 2wiki and hotpotqa settings, variants w = b1w and t = b1t) with B1M's columns, beside
B1M's reads of zrckpn and B1P's reads of zg6r. Nothing trains. zb1m.py, zb1p.py, zgnx.py, zml1s.py, zp1c3.py, b1b.py and b1f.py
are imported and not edited; everything is written under outputs/idpa/.

    python outputs/mp_unified/zidpa.py --selftest
    python outputs/mp_unified/zidpa.py fork
    python outputs/mp_unified/zidpa.py read --name scr-gnx-nx-s0 --variant w       # the host's GPU, the fit's own device
    python outputs/mp_unified/zidpa.py report

fork   GNX's six fits copied to outputs/idpa/fits/<v>/<fit> without their reads (reads/, read.json); models.pt checked.
read   zml1s.run_arm(read) under zgnx.on_gnx (GNX's names, stamp key, model and carve patches, p@swa alone) with B1P's patches
       for the roots, the gate, the eval carves and the read flags (as zb1p.read), and zgnx.on_split given B1M's column folder
       (outputs/b1m/ltm, read only). No ML1 or GNX folder is written (no zml1s.stamp); idpa_read.json lands beside the reads.
report per setting, variant and domain: the 3-seed R@5 of GNX, zrckpn and zg6r with 95% intervals, GNX minus each peer with its
       interval, a label by the interval and the number of fit pairs where GNX is above; the b1w in-domain gap and calls against
       the best published rows (as B1M): outputs/idpa/report.json and report.md.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zb1m as B1M  # noqa: E402  (B1M's settings, carves, roots, gate, column folder and zrckpn's reads; read only)
import zgnx as GX  # noqa: E402  (GNX's fits, patches and column loader; read only)

M1 = GX.M1
WQ = M1.WQ
DECLARED = "docs/IDP_IDENTICAL_INPUT_PAIR.md"
STAGE = "IDPA"
OUT = ROOT / "outputs" / "idpa"
SCREEN_FITS = ROOT / "outputs" / "screen" / "fits"
SETTINGS, VARIANTS, CARVE = B1M.SETTINGS, B1M.VARIANTS, B1M.CARVE
SPLITS, SEEDS, IN_DOMAIN = B1M.SPLITS, B1M.SEEDS, B1M.IN_DOMAIN
FITS = dict(GX.FITS)                                    # GNX's six fits: name -> (arm, seed, split)
GNX, MLP, GNN = "gnx", B1M.MLP, B1M.GNN
ARMS = (GNX, MLP, GNN)
GNX_ABOVE, OTHER_ABOVE, LEVEL = "GNX_ABOVE", "OTHER_ABOVE", "LEVEL"
assert len(FITS) == 6 and set(sd for _a, sd, _s in FITS.values()) == set(SEEDS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zidpa: {msg}", flush=True)


utc, sha_file, sha_src, rel, read_json, write_json = (B1M.utc, B1M.sha_file, B1M.sha_src, B1M.rel, B1M.read_json,
                                                      B1M.write_json)


def fit_of(arm, sd, sp):
    return GX.fit_name(sd, sp) if arm == GNX else B1M.fit_name(arm, sd, sp)


def read_path(arm, fit, v, ds):
    root = {GNX: OUT, MLP: B1M.OUT, GNN: B1M.B1P}[arm]
    return root / "fits" / v / fit / "reads" / f"{ds}__{CARVE[v]}.npz"


# ── the forks and the reads ──────────────────────────────────────────────────


def fork():
    """GNX's six fits, once per variant, without their reads; models.pt checked."""
    for v in VARIANTS:
        for fit in FITS:
            src, dst = SCREEN_FITS / fit, OUT / "fits" / v / fit
            if not (src / "models.pt").exists():
                raise SystemExit(f"IDPA: {rel(src)} holds no models.pt (GNX's fit is not filed here)")
            want = sha_file(src / "models.pt")
            if (dst / "fork.json").exists():
                if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                    raise SystemExit(f"IDPA: {rel(dst)} holds other models than {rel(src)}")
                continue
            tmp = dst.with_name(dst.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, tmp, ignore=B1M.fork_ignore(src))
            if sha_file(tmp / "models.pt") != want:
                raise SystemExit(f"IDPA: {rel(tmp)}'s models.pt is not the source's")
            os.replace(tmp, dst)
            write_json(dst / "fork.json", {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "variant": v,
                                           "source": rel(src), "models_sha256": want, "zidpa_sha256": sha_src(__file__),
                                           "utc": utc()})
            log(f"fork {rel(src)} -> {rel(dst)}")
    return 0


def read_flags(v):
    return ["--read", ",".join(f"{ds}={CARVE[v]}" for ds in SETTINGS), "--out-root", rel(OUT / "fits" / v)]


def guard_read(orig, flags):
    """rmatch's read with this stage's --read and --out-root: added when absent, refused when other."""
    def rm_read(argv):
        argv = list(argv)
        for i in (0, 2):
            if flags[i] not in argv:
                argv += flags[i:i + 2]
        if argv[argv.index("--out-root") + 1] != flags[3] or argv[argv.index("--read") + 1] != flags[1]:
            raise SystemExit(f"IDPA: the read's flags are not IDPA's: {argv}")
        return orig(argv)
    return rm_read


def read(fit, v):
    """GNX's read of a fork on B1P's roots, gate and carves with B1M's columns (zml1s.run_arm under zgnx.on_gnx, no stamp)."""
    import zp1c3 as Z3
    if fit not in FITS:
        raise SystemExit(f"IDPA: {fit} is not one of GNX's fits ({', '.join(FITS)})")
    _arm, _seed, split = FITS[fit]
    carve = CARVE[v]
    fdir = OUT / "fits" / v / fit
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"IDPA: {rel(fdir)} is not forked (zidpa.py fork)")
    outs = [fdir / "reads" / f"{ds}__{carve}.npz" for ds in SETTINGS]
    if (fdir / "idpa_read.json").exists() and all(p.exists() for p in outs):
        log(f"read {fit} {v}: filed")
        return 0
    G = GX.import_g()
    call = GX.DEP.iso_call()
    flags = read_flags(v)
    roots = {**M1.ROOTS, **B1M.READ_ROOTS}
    t = time.time()
    rc, extra, nxt = 1, {}, None
    with GX.on_gnx(G, call):
        env = M1.device_env()
        M1.where_ok(env, "read")
        _rec, k = M1.fit_stamp(fit)
        if M1.device_of(env) != k.get("device"):
            raise SystemExit(f"IDPA: {fit} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
        nk = k.get("next") or {}
        if (nk.get("split"), nk.get("columns")) != (split, list(GX.LTMC.COLS)) or not nk.get("fit_sha256"):
            raise SystemExit(f"IDPA: {fit} was not trained on LTM's columns of {split} ({nk})")
        GX.LOADED.clear()
        try:
            with WQ.patch(M1, ROOTS=roots, GATE=B1M.GATE), \
                    WQ.patch(Z3, eval_carves=lambda: [(ds, carve) for ds in SETTINGS]), \
                    WQ.patch(Z3.RM, read=guard_read(Z3.RM.read, flags)), \
                    GX.on_split(split, nk["fit_sha256"], out=str(B1M.LTM_OUT)):
                rc, extra = M1.run_arm(fit, env, "read")
                nxt = GX.next_block(split)
        finally:
            GX.LOADED.clear()
    if rc:
        return rc
    miss = sorted({f"{ds}/{carve}" for ds in SETTINGS} - set(nxt["carves"]))
    if miss:
        raise SystemExit(f"IDPA: {fit}'s read did not load the columns of {miss}")
    if nxt["fit_sha256"] != nk["fit_sha256"]:
        raise SystemExit(f"IDPA: {fit}'s read loaded columns of the scorer {str(nxt['fit_sha256'])[:12]}, not its own "
                         f"{nk['fit_sha256'][:12]}")
    gone = [rel(p) for p in outs if not p.exists()]
    if gone:
        raise SystemExit(f"IDPA: the read wrote no {gone}")
    write_json(fdir / "idpa_read.json", {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "arm": GX.ARM, "variant": v,
                                         "carve": carve, "split": split, "reads": {rel(p): sha_file(p) for p in outs},
                                         "device": M1.device_of(env), "env": env, "gate_sha256": sha_file(ROOT / B1M.GATE),
                                         "next": nxt, "extra": json.loads(json.dumps(extra, default=str)),
                                         "seconds": round(time.time() - t, 1), "zidpa_sha256": sha_src(__file__),
                                         "utc": utc()})
    log(f"read {fit} {v}: {time.time() - t:.0f} s")
    return 0


# ── the report ───────────────────────────────────────────────────────────────


def label(ci):
    """GNX minus its peer, by the 95% interval of the paired per-question difference."""
    return GNX_ABOVE if ci[0] > 0 else OTHER_ABOVE if ci[1] < 0 else LEVEL


def r100(x):
    return round(100 * float(x), 2)


def pair_stats(BB, a, b):
    """a minus b over the questions: the mean difference, its 95% interval and the label."""
    d = a - b
    ci = BB.boot_ci(d)
    return {"R@5": round(float(d.mean()), 4), "ci": ci, "label": label(ci)}


def report():
    import b1b as BB
    import b1f as B1F
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in IN_DOMAIN.items()}, "settings": {}, "missing": []}
    for ds in SETTINGS:
        qs = read_json(B1M.BENCH / ds / "queries.json")
        gc = [len(q["golds"]) for q in qs]
        s = {"questions": len(qs), "fits": {}, "in_domain": {}, "zero_shot": {}, "gnx_minus_mlp": {}, "gnx_minus_gnn": {},
             "carry_over": {}}
        perq = {}
        for v in VARIANTS:
            for arm in ARMS:
                for sp in SPLITS:
                    for sd in SEEDS:
                        fit = fit_of(arm, sd, sp)
                        p = read_path(arm, fit, v, ds)
                        if not p.exists():
                            rec["missing"].append(rel(p))
                            continue
                        with np.load(p) as z:
                            m = BB.per_question(z, gc)
                        perq[(arm, v, fit)] = m
                        s["fits"][f"{arm}/{CARVE[v]}/{fit}"] = {**{k: round(float(x.mean()), 4) for k, x in m.items()},
                                                                "file_sha256": sha_file(p)}
        means = {}
        for which, splits in (("in_domain", IN_DOMAIN[ds]), ("zero_shot", tuple(x for x in SPLITS if x not in IN_DOMAIN[ds]))):
            if not splits:
                continue
            for v in VARIANTS:
                for arm in ARMS:
                    fits = [fit_of(arm, sd, sp) for sp in splits for sd in SEEDS]
                    got = [perq[(arm, v, f)] for f in fits if (arm, v, f) in perq]
                    if len(got) != len(fits):
                        continue
                    mean = {k: np.mean(np.stack([g[k] for g in got]), axis=0) for k in ("R@5", "FC@5", "hit@1")}
                    means[(arm, v, which)] = mean["R@5"]
                    s[which][f"{arm}/{CARVE[v]}"] = {"R@5": round(float(mean["R@5"].mean()), 4),
                                                     "R5_ci": BB.boot_ci(mean["R@5"]),
                                                     "FC@5": round(float(mean["FC@5"].mean()), 4),
                                                     "hit@1": round(float(mean["hit@1"].mean()), 4), "fits": len(got),
                                                     "per_fit": [round(float(g["R@5"].mean()), 4) for g in got]}
                for peer, key in ((MLP, "gnx_minus_mlp"), (GNN, "gnx_minus_gnn")):
                    if (GNX, v, which) not in means or (peer, v, which) not in means:
                        continue
                    st = pair_stats(BB, means[(GNX, v, which)], means[(peer, v, which)])
                    pairs = [{"gnx": fit_of(GNX, sd, sp), "peer": fit_of(peer, sd, sp),
                              "R@5": round(float((perq[(GNX, v, fit_of(GNX, sd, sp))]["R@5"]
                                                  - perq[(peer, v, fit_of(peer, sd, sp))]["R@5"]).mean()), 4)}
                             for sp in splits for sd in SEEDS]
                    s[key][f"{which}/{CARVE[v]}"] = {**st, "fit_pairs_gnx_above": sum(1 for x in pairs if x["R@5"] > 0),
                                                     "fit_pairs": len(pairs), "per_fit_pair": pairs}
            for arm in ARMS:
                if (arm, "w", which) in means and (arm, "t", which) in means:
                    d = means[(arm, "w", which)] - means[(arm, "t", which)]
                    s["carry_over"][f"{arm}/{which}"] = {"R@5": round(float(d.mean()), 4), "ci": BB.boot_ci(d)}
        s["best_published"] = {}
        for which, peer_only in (("any", False), ("peer_reviewed", True)):
            val, row = B1F.best_published(ds, peer_only)
            s["best_published"][which] = {"row": row, "R@5": val}
        s["calls"], s["gap_points"] = {}, {}
        for arm in ARMS:
            c = s["in_domain"].get(f"{arm}/b1w")
            if c is None:
                continue
            s["calls"][f"{arm}/b1w"] = {w: BB.call(c["R5_ci"], b["R@5"] / 100) for w, b in s["best_published"].items()}
            s["gap_points"][f"{arm}/b1w"] = {w: round(100 * c["R@5"] - b["R@5"], 2) for w, b in s["best_published"].items()}
        rec["settings"][ds] = s
    rec["zidpa_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "report.json", rec)
    write_md(rec)
    log(f"report: {rel(OUT / 'report.md')}; missing {len(rec['missing'])}")
    return 0


def fmt(c):
    return "--" if c is None else f"{r100(c['R@5']):.1f}"


def write_md(rec):
    L = ["# IDPA: GNX (the same inputs as the MLP of record) on HippoRAG 2's settings, beside zrckpn and zg6r", "",
         f"Declared in {DECLARED} section 5. R@5 in points, the 3-seed mean over the splits of the domain (95% interval of the "
         f"questions, {rec['boot']['resamples']} resamples, seed {rec['boot']['seed']}). Differences are GNX minus the peer, "
         f"with the interval of the paired per-question difference, a label by it, and the fit pairs where GNX is above. "
         f"No threshold, no adoption call.", "",
         "| setting | domain | carve | zrckpn | zg6r | GNX | GNX - zrckpn | label | pairs above | GNX - zg6r | label | pairs above |",
         "|---|---|---|---:|---:|---:|---|---|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for which in ("in_domain", "zero_shot"):
            for cv in ("b1w", "b1t"):
                g = s[which].get(f"{GNX}/{cv}")
                if g is None:
                    continue
                cells = []
                for key in ("gnx_minus_mlp", "gnx_minus_gnn"):
                    d = s[key].get(f"{which}/{cv}")
                    cells += ["--", "--", "--"] if d is None else [
                        f"{r100(d['R@5']):+.1f} ({r100(d['ci'][0]):+.1f}, {r100(d['ci'][1]):+.1f})", d["label"],
                        f"{d['fit_pairs_gnx_above']}/{d['fit_pairs']}"]
                L.append(f"| {ds} | {which.replace('_', '-')} | {cv} | {fmt(s[which].get(f'{MLP}/{cv}'))} | "
                         f"{fmt(s[which].get(f'{GNN}/{cv}'))} | {fmt(g)} | " + " | ".join(cells) + " |")
    L += ["", "## b1w in-domain against the best published rows", "",
          "| setting | arm | R@5 (95% interval) | gap to best peer-reviewed | call | gap to best any | call |", "|---|---|---|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for arm in ARMS:
            c = s["in_domain"].get(f"{arm}/b1w")
            if c is None:
                continue
            gp, cl = s["gap_points"][f"{arm}/b1w"], s["calls"][f"{arm}/b1w"]
            L.append(f"| {ds} | {arm} | {r100(c['R@5']):.1f} ({r100(c['R5_ci'][0]):.1f}, {r100(c['R5_ci'][1]):.1f}) | "
                     f"{gp['peer_reviewed']:+.1f} | {cl['peer_reviewed']} | {gp['any']:+.1f} | {cl['any']} |")
    if rec["missing"]:
        L += ["", f"Missing reads: {len(rec['missing'])}."] + [f"- {m}" for m in rec["missing"][:12]]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    # 1. the names: GNX's six fits, B1M's settings and splits, the paths of the three arms' reads
    assert sorted(FITS) == sorted(GX.fit_name(sd, sp) for sd in SEEDS for sp in SPLITS), "GNX's fits"
    assert FITS["scr-gnx-nx-s0"][2] == "WQ-musique" and FITS["scr-gnx-nx-s2-hp"][2] == "WQ-hotpotqa", "splits"
    assert fit_of(MLP, 1, "WQ-hotpotqa") == "scr-ltm-zrckpn-s1-hp" and fit_of(GNN, 0, "WQ-musique") == "scr-ml1-zg6r-s0"
    assert read_path(GNX, "scr-gnx-nx-s0", "w", "2wiki") == OUT / "fits/w/scr-gnx-nx-s0/reads/2wiki__b1w.npz"
    assert read_path(MLP, "scr-ltm-zrckpn-s0", "t", "musique") == B1M.OUT / "fits/t/scr-ltm-zrckpn-s0/reads/musique__b1t.npz"
    assert read_path(GNN, "scr-ml1-zg6r-s0-hp", "t", "musique") == B1M.B1P / "fits/t/scr-ml1-zg6r-s0-hp/reads/musique__b1t.npz"
    assert read_flags("t")[1] == "musique=b1t,2wiki=b1t,hotpotqa=b1t" and read_flags("w")[3] == "outputs/idpa/fits/w"
    # 2. the read guard: flags added when absent, refused when other
    got = []
    g = guard_read(lambda a: got.append(a) or 0, read_flags("w"))
    assert g(["--name", "x"]) == 0 and got[-1][-4:] == read_flags("w"), "flags added"
    for bad in (["--name", "x", "--read", "musique=b1w", "--out-root", "elsewhere"],
                ["--name", "x", "--read", "2wiki=b1w", "--out-root", "outputs/idpa/fits/w"]):
        try:
            g(bad)
            raise AssertionError("other flags passed")
        except SystemExit:
            pass
    # 3. the label
    assert (label([0.001, 0.02]), label([-0.02, -0.001]), label([-0.01, 0.01]), label([0.0, 0.01])) == (
        GNX_ABOVE, OTHER_ABOVE, LEVEL, LEVEL), "label"
    # 4. the fork skips only the source's own reads (B1M's rule) and keeps models.pt
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "src"
        (src / "reads").mkdir(parents=True)
        for f in ("models.pt", "read.json", "screen.json", "reads/a.npz"):
            (src / f).write_bytes(b"x")
        shutil.copytree(src, Path(t) / "dst", ignore=B1M.fork_ignore(src))
        kept = sorted(p.relative_to(Path(t) / "dst").as_posix() for p in (Path(t) / "dst").rglob("*") if p.is_file())
        assert kept == ["models.pt", "screen.json"], kept
    # 5. GNX's patch points the read needs exist (a rename in zgnx or zml1s stops here, not on the host)
    for name in ("on_gnx", "on_split", "next_block", "import_g", "fit_name", "LOADED", "DEP", "LTMC", "ARM"):
        assert hasattr(GX, name), f"zgnx.{name}"
    for name in ("run_arm", "device_env", "where_ok", "fit_stamp", "device_of", "ROOTS", "GATE"):
        assert hasattr(M1, name), f"zml1s.{name}"
    assert B1M.READ_ROOTS["cache"] == "outputs/b1p/cache" and B1M.GATE == "outputs/b1p/gate.json"
    assert str(B1M.LTM_OUT).replace("\\", "/").endswith("outputs/b1m/ltm")
    # 6. the report's pieces on synthetic reads: the paired difference, its label, the counts
    class _BB:
        CAND, BOOT, SEED = "p@swa", 200, 0

        @staticmethod
        def boot_ci(x, seed=0, n=200):
            rng = np.random.default_rng(seed)
            m = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n)])
            return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]
    a, b = np.full(400, 0.8), np.full(400, 0.7)
    s = pair_stats(_BB, a, b)
    assert s["label"] == GNX_ABOVE and abs(s["R@5"] - 0.1) < 1e-9, s
    rng = np.random.default_rng(1)
    c = rng.random(400)
    assert pair_stats(_BB, c, c)["label"] == LEVEL, "equal reads are LEVEL"
    print("zidpa self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fork", "read", "report", "selftest"))
    ap.add_argument("--name")
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return selftest()
    if a.cmd == "read":
        if a.name is None or a.variant is None:
            raise SystemExit("read: --name and --variant are required")
        return read(a.name, a.variant)
    if a.cmd == "fork":
        return fork()
    if a.cmd == "report":
        return report()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
