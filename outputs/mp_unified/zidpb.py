"""IDP amendment A2 (docs/SWP_S_TO_R_SWAP.md section 3): GNX, the GNN that reads the MLP of record's inputs, read on b1, B1a's own
carve of HippoRAG 2's three settings (2wiki, hotpotqa: the corpus hyperlinks, the dataset's own S; musique: the package's
title-mention graph), with B1H's columns, so that the same-input pair (GNX, zrckpn) has the whole ladder b1t -> b1w -> b1 beside
zg6r. Nothing trains. zidpa.py, zb1h.py, zb1m.py, zb1p.py, zgnx.py, zml1s.py and zp1c3.py are imported and not edited;
everything is written under outputs/idpb/.

    python outputs/mp_unified/zidpb.py --selftest
    python outputs/mp_unified/zidpb.py fork
    python outputs/mp_unified/zidpb.py read --name scr-gnx-nx-s0 --tag main|musique     # the host's GPU, the fit's own device
    python outputs/mp_unified/zidpb.py report                                           # the laptop

fork   GNX's six fits copied to outputs/idpb/fits/h/<fit> without their reads; models.pt checked (as zidpa.fork).
read   zml1s.run_arm(read) under zgnx.on_gnx with B1H's roots, gate (main: 2wiki and hotpotqa; musique: musique), carve b1 and
       B1H's column folder (outputs/b1h/ltm, read only), the flags as zidpa.read. idpb_read_<tag>.json lands beside the reads.
report per setting and domain: R@5 of GNX, zrckpn and zg6r on b1t, b1w and b1 (3-seed means, 95% intervals), the paired steps
       b1w-b1t, b1-b1w, b1-b1t, the share of the b1 gain at b1w, GNX minus zrckpn on each carve and how that gap changes from
       b1t to b1w and to b1: outputs/idpb/report.json and report.md. GNX's b1t and b1w reads are IDPA's (outputs/idpa), zrckpn's
       and zg6r's are B1M's, B1P's and B1H's, all read only.
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
import zidpa as A  # noqa: E402  (IDP A1: GNX's fits and the read's shape; read only)
import zb1h as H  # noqa: E402  (B1H: roots, gate, carve b1, columns and the two peers' reads; read only)

GX, B1M, M1 = A.GX, A.B1M, A.M1
WQ = M1.WQ
DECLARED = "docs/SWP_S_TO_R_SWAP.md"
STAGE = "IDPB"
OUT = ROOT / "outputs" / "idpb"
CARVE, CARVES, TAGS = H.CARVE, H.CARVES, H.TAGS          # "b1"; ("b1t", "b1w", "b1"); {"main": (2wiki, hotpotqa), "musique": (musique,)}
SETTINGS, SPLITS, SEEDS, IN_DOMAIN = H.SETTINGS, H.SPLITS, H.SEEDS, H.IN_DOMAIN
FITS = dict(GX.FITS)                                     # GNX's six fits: name -> (arm, seed, split)
GNX, MLP, GNN = "gnx", H.MLP, H.GNN
ARMS = (GNX, MLP, GNN)
assert len(FITS) == 6 and OUT != A.OUT


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zidpb: {msg}", flush=True)


utc, sha_file, sha_src, rel, read_json, write_json = A.utc, A.sha_file, A.sha_src, A.rel, A.read_json, A.write_json


def fits_for(arm, splits):
    """The fit names of an arm on the given splits, seeds inside."""
    if arm == GNX:
        return [GX.fit_name(sd, sp) for sp in splits for sd in SEEDS]
    return H.fits_of(arm, splits)


def read_path(arm, fit, carve, ds):
    """Where an arm's read of a fit on a carve lives: GNX's b1t and b1w are IDPA's, its b1 is this stage's; the peers' are B1H's."""
    if arm == GNX:
        root = OUT / "fits" / "h" if carve == CARVE else A.OUT / "fits" / ("w" if carve == "b1w" else "t")
        return root / fit / "reads" / f"{ds}__{carve}.npz"
    return H.read_path(arm, fit, carve, ds)


# ── the fork and the read ────────────────────────────────────────────────────


def fork():
    """GNX's six fits copied without their reads; models.pt checked against the source."""
    for fit in FITS:
        src, dst = A.SCREEN_FITS / fit, OUT / "fits" / "h" / fit
        if not (src / "models.pt").exists():
            raise SystemExit(f"IDPB: {rel(src)} holds no models.pt (GNX's fit is not filed here)")
        want = sha_file(src / "models.pt")
        if (dst / "fork.json").exists():
            if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                raise SystemExit(f"IDPB: {rel(dst)} holds other models than {rel(src)}")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp, ignore=B1M.fork_ignore(src))
        if sha_file(tmp / "models.pt") != want:
            raise SystemExit(f"IDPB: {rel(tmp)}'s models.pt is not the source's")
        os.replace(tmp, dst)
        write_json(dst / "fork.json", {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "source": rel(src),
                                       "models_sha256": want, "zidpb_sha256": sha_src(__file__), "utc": utc()})
        log(f"fork {rel(src)} -> {rel(dst)}")
    return 0


def read_flags(tag):
    return ["--read", ",".join(f"{ds}={CARVE}" for ds in TAGS[tag]), "--out-root", rel(OUT / "fits" / "h")]


def read(fit, tag):
    """GNX's read of a fork on B1H's roots, gate and carve b1 with B1H's columns (zidpa.read's body with B1H's pieces)."""
    import zp1c3 as Z3
    if fit not in FITS:
        raise SystemExit(f"IDPB: {fit} is not one of GNX's fits ({', '.join(FITS)})")
    if tag not in TAGS:
        raise SystemExit(f"IDPB: --tag is one of {', '.join(TAGS)}")
    _arm, _seed, split = FITS[fit]
    sets = TAGS[tag]
    fdir = OUT / "fits" / "h" / fit
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"IDPB: {rel(fdir)} is not forked (zidpb.py fork)")
    outs = [fdir / "reads" / f"{ds}__{CARVE}.npz" for ds in sets]
    marker = fdir / f"idpb_read_{tag}.json"
    if marker.exists() and all(p.exists() for p in outs):
        log(f"read {fit} {tag}: filed")
        return 0
    G = GX.import_g()
    call = GX.DEP.iso_call()
    flags = read_flags(tag)
    roots = {**M1.ROOTS, **H.read_roots()}
    gate = rel(H.gate_path(tag))
    t = time.time()
    rc, extra, nxt = 1, {}, None
    with GX.on_gnx(G, call):
        env = M1.device_env()
        M1.where_ok(env, "read")
        _rec, k = M1.fit_stamp(fit)
        if M1.device_of(env) != k.get("device"):
            raise SystemExit(f"IDPB: {fit} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
        nk = k.get("next") or {}
        if (nk.get("split"), nk.get("columns")) != (split, list(GX.LTMC.COLS)) or not nk.get("fit_sha256"):
            raise SystemExit(f"IDPB: {fit} was not trained on LTM's columns of {split} ({nk})")
        GX.LOADED.clear()
        try:
            with WQ.patch(M1, ROOTS=roots, GATE=gate), \
                    WQ.patch(Z3, eval_carves=lambda: [(ds, CARVE) for ds in sets]), \
                    WQ.patch(Z3.RM, read=A.guard_read(Z3.RM.read, flags)), \
                    GX.on_split(split, nk["fit_sha256"], out=str(H.LTM_OUT)):
                rc, extra = M1.run_arm(fit, env, "read")
                nxt = GX.next_block(split)
        finally:
            GX.LOADED.clear()
    if rc:
        return rc
    miss = sorted({f"{ds}/{CARVE}" for ds in sets} - set(nxt["carves"]))
    if miss:
        raise SystemExit(f"IDPB: {fit}'s read did not load the columns of {miss}")
    if nxt["fit_sha256"] != nk["fit_sha256"]:
        raise SystemExit(f"IDPB: {fit}'s read loaded columns of the scorer {str(nxt['fit_sha256'])[:12]}, not its own "
                         f"{nk['fit_sha256'][:12]}")
    gone = [rel(p) for p in outs if not p.exists()]
    if gone:
        raise SystemExit(f"IDPB: the read wrote no {gone}")
    write_json(marker, {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "arm": GX.ARM, "tag": tag, "carve": CARVE,
                        "split": split, "reads": {rel(p): sha_file(p) for p in outs}, "device": M1.device_of(env), "env": env,
                        "gate_sha256": sha_file(ROOT / gate), "next": nxt,
                        "extra": json.loads(json.dumps(extra, default=str)), "seconds": round(time.time() - t, 1),
                        "zidpb_sha256": sha_src(__file__), "utc": utc()})
    log(f"read {fit} {tag}: {time.time() - t:.0f} s")
    return 0


# ── the report ───────────────────────────────────────────────────────────────


def r100(x):
    return round(100 * float(x), 2)


def report():
    import b1b as BB
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in IN_DOMAIN.items()}, "settings": {}, "missing": []}
    for ds in SETTINGS:
        qs = read_json(B1M.BENCH / ds / "queries.json")
        gc = [len(q["golds"]) for q in qs]
        perq = {}
        for carve in CARVES:
            for arm in ARMS:
                for fit in fits_for(arm, SPLITS):
                    p = read_path(arm, fit, carve, ds)
                    if not p.exists():
                        rec["missing"].append(rel(p))
                        continue
                    with np.load(p) as z:
                        perq[(arm, carve, fit)] = BB.per_question(z, gc)
        s = {"questions": len(qs), "in_domain": {}, "zero_shot": {}, "paired": {}, "paired_zero_shot": {},
             "share_of_b1_gain_at_b1w": {}, "gnx_minus_mlp": {}, "gap_change": {}}
        means = {}
        for which, splits in (("in_domain", IN_DOMAIN[ds]), ("zero_shot", tuple(x for x in SPLITS if x not in IN_DOMAIN[ds]))):
            if not splits:
                continue
            for carve in CARVES:
                for arm in ARMS:
                    fits = fits_for(arm, splits)
                    got = [perq[(arm, carve, f)] for f in fits if (arm, carve, f) in perq]
                    if len(got) != len(fits):
                        continue
                    mean = np.mean(np.stack([g["R@5"] for g in got]), axis=0)
                    means[(arm, carve, which)] = mean
                    s[which][f"{arm}/{carve}"] = {"R@5": round(float(mean.mean()), 4), "R5_ci": BB.boot_ci(mean),
                                                  "fits": len(got), "per_fit": [round(float(g["R@5"].mean()), 4) for g in got]}
            pair = s["paired" if which == "in_domain" else "paired_zero_shot"]
            for arm in ARMS:
                t, w, h = (means.get((arm, c, which)) for c in CARVES)
                if t is None or w is None or h is None:
                    continue
                pair[f"b1w-b1t {arm}"] = H.ci_of(w - t)
                pair[f"b1-b1w {arm}"] = H.ci_of(h - w)
                pair[f"b1-b1t {arm}"] = H.ci_of(h - t)
                den = float((h - t).mean())
                if den > 0:
                    s["share_of_b1_gain_at_b1w"][f"{which}/{arm}"] = round(float((w - t).mean()) / den, 3)
            for carve in CARVES:
                g, m = means.get((GNX, carve, which)), means.get((MLP, carve, which))
                if g is None or m is None:
                    continue
                ci = BB.boot_ci(g - m)
                s["gnx_minus_mlp"][f"{which}/{carve}"] = {"R@5": round(float((g - m).mean()), 4), "ci": ci, "label": A.label(ci)}
            gt, mt = means.get((GNX, "b1t", which)), means.get((MLP, "b1t", which))
            for carve in ("b1w", "b1"):
                g, m = means.get((GNX, carve, which)), means.get((MLP, carve, which))
                if None in (g, m, gt, mt):
                    continue
                s["gap_change"][f"{which}/{carve} vs b1t"] = H.ci_of((g - m) - (gt - mt))
        s["questions_note"] = "R@5 of the p@swa candidate, 1,000 questions, no LLM"
        rec["settings"][ds] = s
    rec["zidpb_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "report.json", rec)
    write_md(rec)
    log(f"report: {rel(OUT / 'report.md')}; missing {len(rec['missing'])}")
    return 0


def cell(c):
    return "--" if c is None else f"{r100(c['R@5']):.1f}"


def ci_txt(d):
    return "--" if d is None else f"{r100(d['R@5']):+.1f} ({r100(d['ci'][0]):+.1f}, {r100(d['ci'][1]):+.1f})"


def write_md(rec):
    L = ["# IDPB: the same-input pair on the ladder b1t -> b1w -> b1 (none, rule links, the dataset's own S)", "",
         f"Declared in {DECLARED}. R@5 in points on HippoRAG 2's settings (1,000 questions each), the 3-seed mean over the splits "
         f"of the domain, 95% interval of the questions ({rec['boot']['resamples']} resamples, seed {rec['boot']['seed']}). "
         f"b1t: structural slot empty, NER and kNN kept. b1w: the rule's links (W2_S) in the slot. b1: the dataset's own S "
         f"(2wiki, hotpotqa: corpus hyperlinks; musique: title mentions, itself a rule). The fits were trained with the rule's "
         f"links in the slot, so every b1 row is a read-time swap of the slot, not a model trained on S. No threshold, no "
         f"adoption call.", "",
         "| setting | domain | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w | b1 - b1t | share of the b1 gain at b1w |",
         "|---|---|---|---:|---:|---:|---|---|---|---:|"]
    for ds, s in rec["settings"].items():
        for which, pk in (("in_domain", "paired"), ("zero_shot", "paired_zero_shot")):
            for arm in ARMS:
                if f"{arm}/b1t" not in s[which]:
                    continue
                p = s[pk]
                sh = s["share_of_b1_gain_at_b1w"].get(f"{which}/{arm}")
                L.append(f"| {ds} | {which.replace('_', '-')} | {arm} | " + " | ".join(cell(s[which].get(f"{arm}/{c}")) for c in CARVES)
                         + f" | {ci_txt(p.get(f'b1w-b1t {arm}'))} | {ci_txt(p.get(f'b1-b1w {arm}'))} | {ci_txt(p.get(f'b1-b1t {arm}'))} | "
                         + ("--" if sh is None else f"{sh:.2f}") + " |")
    L += ["", "## Does message passing matter more with structure in? GNX minus zrckpn, same inputs", "",
          "| setting | domain | b1t | b1w | b1 | gap change b1t -> b1w | gap change b1t -> b1 |", "|---|---|---|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for which in ("in_domain", "zero_shot"):
            if f"{which}/b1t" not in s["gnx_minus_mlp"]:
                continue
            g = s["gnx_minus_mlp"]
            L.append(f"| {ds} | {which.replace('_', '-')} | " + " | ".join(
                f"{ci_txt(g.get(f'{which}/{c}'))} {g[f'{which}/{c}']['label'] if f'{which}/{c}' in g else ''}" for c in CARVES)
                + f" | {ci_txt(s['gap_change'].get(f'{which}/b1w vs b1t'))} | {ci_txt(s['gap_change'].get(f'{which}/b1 vs b1t'))} |")
    if rec["missing"]:
        L += ["", f"Missing reads: {len(rec['missing'])}."] + [f"- {m}" for m in rec["missing"][:12]]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    # 1. names and paths
    assert sorted(FITS) == sorted(GX.fit_name(sd, sp) for sd in SEEDS for sp in SPLITS), "GNX's fits"
    assert sorted(ds for t in TAGS.values() for ds in t) == sorted(SETTINGS), "the tags cover the settings once"
    assert CARVES == ("b1t", "b1w", "b1") and CARVE == "b1"
    assert fits_for(GNX, ("WQ-hotpotqa",)) == [f"scr-gnx-nx-s{sd}-hp" for sd in SEEDS], fits_for(GNX, ("WQ-hotpotqa",))
    assert fits_for(MLP, ("WQ-musique",)) == [f"scr-ltm-zrckpn-s{sd}" for sd in SEEDS]
    assert read_path(GNX, "scr-gnx-nx-s0", "b1", "2wiki") == OUT / "fits/h/scr-gnx-nx-s0/reads/2wiki__b1.npz"
    assert read_path(GNX, "scr-gnx-nx-s0", "b1w", "2wiki") == A.OUT / "fits/w/scr-gnx-nx-s0/reads/2wiki__b1w.npz"
    assert read_path(GNX, "scr-gnx-nx-s0", "b1t", "2wiki") == A.OUT / "fits/t/scr-gnx-nx-s0/reads/2wiki__b1t.npz"
    assert read_path(MLP, "scr-ltm-zrckpn-s0", "b1", "musique") == H.OUT / "fits/h/scr-ltm-zrckpn-s0/reads/musique__b1.npz"
    assert read_path(GNN, "scr-ml1-zg6r-s0", "b1t", "musique") == H.B1P / "fits/t/scr-ml1-zg6r-s0/reads/musique__b1t.npz"
    assert read_flags("main") == ["--read", "2wiki=b1,hotpotqa=b1", "--out-root", "outputs/idpb/fits/h"], read_flags("main")
    assert read_flags("musique")[1] == "musique=b1"
    # 2. the read guard (IDPA's): flags added when absent, refused when other
    got = []
    g = A.guard_read(lambda a: got.append(a) or 0, read_flags("main"))
    assert g(["--name", "x"]) == 0 and got[-1][-4:] == read_flags("main"), "flags added"
    for bad in (["--name", "x", "--read", "musique=b1", "--out-root", "elsewhere"],
                ["--name", "x", "--read", "2wiki=b1", "--out-root", "outputs/idpb/fits/h"]):
        try:
            g(bad)
            raise AssertionError("other flags passed")
        except SystemExit:
            pass
    # 3. B1H's pieces the read needs exist and name B1H's folders
    roots = H.read_roots()
    assert roots["pools"] == "outputs/b1h/pools" and roots["cache"] == "outputs/b1h/cache" and roots["kt"] == "outputs/b1h/kt"
    assert str(H.LTM_OUT).replace("\\", "/").endswith("outputs/b1h/ltm")
    assert rel(H.gate_path("main")) == "outputs/b1h/gate.json" and rel(H.gate_path("musique")) == "outputs/b1h/gate_musique.json"
    for name in ("on_gnx", "on_split", "next_block", "import_g", "fit_name", "LOADED", "DEP", "LTMC", "ARM"):
        assert hasattr(GX, name), f"zgnx.{name}"
    for name in ("run_arm", "device_env", "where_ok", "fit_stamp", "device_of", "ROOTS", "GATE"):
        assert hasattr(M1, name), f"zml1s.{name}"
    # 4. the fork keeps models.pt and drops only the source's own reads
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "src"
        (src / "reads").mkdir(parents=True)
        for f in ("models.pt", "read.json", "screen.json", "reads/a.npz"):
            (src / f).write_bytes(b"x")
        shutil.copytree(src, Path(t) / "dst", ignore=B1M.fork_ignore(src))
        kept = sorted(p.relative_to(Path(t) / "dst").as_posix() for p in (Path(t) / "dst").rglob("*") if p.is_file())
        assert kept == ["models.pt", "screen.json"], kept
    # 5. the share and the gap change on synthetic means
    t, w, h = np.full(100, 0.5), np.full(100, 0.8), np.full(100, 0.9)
    assert abs(float((w - t).mean()) / float((h - t).mean()) - 0.75) < 1e-9
    print("zidpb self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fork", "read", "report", "selftest"))
    ap.add_argument("--name")
    ap.add_argument("--tag", choices=tuple(TAGS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return selftest()
    if a.cmd == "read":
        if a.name is None or a.tag is None:
            raise SystemExit("read: --name and --tag are required")
        return read(a.name, a.tag)
    if a.cmd == "fork":
        return fork()
    if a.cmd == "report":
        return report()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
