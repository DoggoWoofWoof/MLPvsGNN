"""B1M (docs/B1M_MLP_OF_RECORD_ON_B1.md): LTM's zrckpn, the MLP of record, read on B1P's six carves of HippoRAG 2's settings,
with LTM's columns built on those carves by LTM's own code and scorers, beside B1P's reads of zg6r (the GNN of record) and
zrckp. Nothing trains. LTM's scorers and fits, and B1P's chain, gate and reads, are read and never written.

    python outputs/mp_unified/zb1m.py --selftest
    python outputs/mp_unified/zb1m.py lists --dataset D --variant w|t
    python outputs/mp_unified/zb1m.py cols --split S --dataset D --variant w|t --host
    python outputs/mp_unified/zb1m.py fork
    python outputs/mp_unified/zb1m.py read --name FIT --variant w|t       # the host's CPU (DOB's device, zltm's threads)
    python outputs/mp_unified/zb1m.py report

lists  ltmcols.lists on the carve, from B1P's cache and links parts (root outputs/b1p), into outputs/b1m/ltm.
cols   ltmcols.cols of the split on the carve, into outputs/b1m/ltm, with col_carves naming the B1 carve and load_folds reading
       LTM's own fold fits (outputs/ltm), both restored after. A B1 carve is no training carve: each question's columns come
       from the fold fit its index draws, as on s1eval.
fork   LTM's six zrckpn fits copied to outputs/b1m/fits/<v>/<fit> without their reads (reads/, read.json); models.pt checked.
read   zltm.read's body under zltm.on_ltm, with zb1p.read's patches: ML1's roots and gate are B1P's, zp1c3's eval carves the
       variant's three B1 carves, rmatch's read takes B1M's --read and --out-root, and zltm.on_split loads B1M's columns. No
       LTM or ML1 folder is written (no zml1s.stamp); b1m_read.json lands beside the reads.
report section 4's tables: outputs/b1m/report.json and .md, and cols.json (each column record's sha256).
"""
import os
import sys

if __name__ == "__main__" and sys.argv[1:2] == ["read"]:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "6"                              # zltm's THREADS, before numpy or torch loads
    os.environ["CUDA_VISIBLE_DEVICES"] = ""               # a CPU read never sees the GPU (zltm's)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import zwq as WQ  # noqa: E402  (stdlib only at import: patch, the splits)

DECLARED = "docs/B1M_MLP_OF_RECORD_ON_B1.md"
STAGE = "B1M"
OUT = ROOT / "outputs" / "b1m"
LTM_OUT = OUT / "ltm"                                   # B1M's lists and columns (ltmcols' layout)
B1P = ROOT / "outputs" / "b1p"                          # B1P's chain, gate and reads: read only
SCREEN_FITS = ROOT / "outputs" / "screen" / "fits"
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
SETTINGS = ("musique", "2wiki", "hotpotqa")
VARIANTS = ("w", "t")
CARVE = {"w": "b1w", "t": "b1t"}
SPLITS = ("WQ-musique", "WQ-hotpotqa")
SUF = {"WQ-musique": "", "WQ-hotpotqa": "-hp"}         # zwq.SUF
TAG = {"WQ-musique": "wqm", "WQ-hotpotqa": "wqh"}
SEEDS = (0, 1, 2)
IN_DOMAIN = {"musique": ("WQ-hotpotqa",), "hotpotqa": ("WQ-musique",), "2wiki": ("WQ-musique", "WQ-hotpotqa")}   # B1P's
GNN, MLP, OLD = "zg6r", "zrckpn", "zrckp"               # the GNN and the MLP of record; zrckp (B1P's MLP) beside
ARMS = (GNN, MLP, OLD)
FITS = {f"scr-ltm-{MLP}-s{sd}{SUF[sp]}": (sd, sp) for sd in SEEDS for sp in SPLITS}
READ_ROOTS = {"pools": "outputs/b1p/pools", "plook": "outputs/b1p/look", "cache": "outputs/b1p/cache",
              "rmatch": "outputs/b1p/chains/rmatch", "zrc": "outputs/b1p/chains/zrc", "links": "outputs/b1p/links",
              "kt": "outputs/b1p/kt"}                   # zb1p.read's roots
GATE = "outputs/b1p/gate.json"                          # B1P's PKO gate (READY)
GNN_ABOVE, MLP_ABOVE, LEVEL = "GNN_ABOVE", "MLP_ABOVE", "LEVEL"
assert len(FITS) == 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zb1m: {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def fit_name(arm, seed, split):
    """zrckpn's fits are LTM's; zg6r's and zrckp's are ML1's, as B1P forked them."""
    return f"scr-ltm-{arm}-s{seed}{SUF[split]}" if arm == MLP else f"scr-ml1-{arm}-s{seed}{SUF[split]}"


def read_path(arm, fit, v, ds):
    return (OUT if arm == MLP else B1P) / "fits" / v / fit / "reads" / f"{ds}__{CARVE[v]}.npz"


# ── LTM's lists and columns on B1P's carves ──────────────────────────────────


def lists(ds, v):
    import ltmcols as LTMC
    LTMC.lists(ds, CARVE[v], root=B1P, out=LTM_OUT)
    return 0


@contextlib.contextmanager
def on_b1_cols(ds, carve):
    """ltmcols with col_carves naming the B1 carve beside the split's own, and load_folds reading LTM's fold fits wherever
    the columns are written; both restored after, also on a failure."""
    import ltmcols as LTMC
    orig_cc, orig_lf = LTMC.col_carves, LTMC.load_folds

    def col_carves(split):
        return tuple(orig_cc(split)) + ((ds, carve),)

    def load_folds(_out, split):
        return orig_lf(LTMC.OUT, split)

    with WQ.patch(LTMC, col_carves=col_carves, load_folds=load_folds):
        yield LTMC


def cols(split, ds, v, host=False):
    import ltmcols as LTMC
    if host:
        import lean_host as LH
        LH.substitute()                                 # the verified mirror in place of the package (ltmcols --host)
    with on_b1_cols(ds, CARVE[v]):
        LTMC.cols(split, ds, CARVE[v], root=B1P, out=LTM_OUT, host=host)
    return 0


# ── the forks and the reads ──────────────────────────────────────────────────


def fork_ignore(src):
    """copytree's ignore: the source folder's own reads/ and read.json, nothing deeper."""
    src = Path(src)
    return lambda d, names: [x for x in names if Path(d) == src and x in ("reads", "read.json")]


def fork():
    """LTM's six zrckpn fits, once per variant, without their reads; models.pt checked."""
    for v in VARIANTS:
        for fit in FITS:
            src, dst = SCREEN_FITS / fit, OUT / "fits" / v / fit
            want = sha_file(src / "models.pt")
            if (dst / "fork.json").exists():
                if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                    raise SystemExit(f"B1M: {rel(dst)} holds other models than {rel(src)}")
                continue
            tmp = dst.with_name(dst.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp, ignore=fork_ignore(src))
            if sha_file(tmp / "models.pt") != want:
                raise SystemExit(f"B1M: {rel(tmp)}'s models.pt is not the source's")
            os.replace(tmp, dst)
            write_json(dst / "fork.json", {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "variant": v,
                                           "source": rel(src), "models_sha256": want, "zb1m_sha256": sha_src(__file__),
                                           "utc": utc()})
            log(f"fork {rel(src)} -> {rel(dst)}")
    return 0


def read_flags(v):
    return ["--read", ",".join(f"{ds}={CARVE[v]}" for ds in SETTINGS), "--out-root", rel(OUT / "fits" / v)]


def guard_read(orig, flags):
    """rmatch's read with B1M's --read and --out-root: added when absent, refused when other."""
    def rm_read(argv):
        argv = list(argv)
        for i in (0, 2):
            if flags[i] not in argv:
                argv += flags[i:i + 2]
        if argv[argv.index("--out-root") + 1] != flags[3] or argv[argv.index("--read") + 1] != flags[1]:
            raise SystemExit(f"B1M: the read's flags are not B1M's: {argv}")
        return orig(argv)
    return rm_read


def read(fit, v):
    """zrckpn's read of a fork on B1P's roots, gate and carves with B1M's columns (zltm.read's body, no stamp)."""
    import zltm as LTM
    import zml1s as M1
    import zp1c3 as Z3
    import ltmcols as LTMC
    if fit not in FITS or LTM.FITS.get(fit) != (MLP, *FITS[fit]):
        raise SystemExit(f"B1M: {fit} is not one of LTM's zrckpn fits")
    _arm, _seed, split = LTM.FITS[fit]
    carve = CARVE[v]
    fdir = OUT / "fits" / v / fit
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"B1M: {rel(fdir)} is not forked (zb1m.py fork)")
    outs = [fdir / "reads" / f"{ds}__{carve}.npz" for ds in SETTINGS]
    if (fdir / "b1m_read.json").exists() and all(p.exists() for p in outs):
        log(f"read {fit} {v}: filed")
        return 0
    Z3P = LTM.import_arm()                              # outside zwq.on_wq, as zltm.fit
    flags = read_flags(v)
    t = time.time()
    rc, nxt = 1, None
    with LTM.on_ltm():
        env = M1.device_env()                           # DOB's cpu_env under on_ltm
        M1.where_ok(env, "read")
        _rec, k = M1.fit_stamp(fit)
        if M1.device_of(env) != k.get("device"):
            raise SystemExit(f"B1M: {fit} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
        nk = k.get("next") or {}
        if (nk.get("declared_in"), nk.get("split"), nk.get("columns")) != (LTM.DECLARED, split, list(LTMC.COLS)) or \
                not nk.get("fit_sha256"):
            raise SystemExit(f"B1M: {fit} was not trained on LTM's columns of {split} ({nk})")
        roots = {**M1.ROOTS, **READ_ROOTS}
        argv = ["--name", fit, "--device", "cpu", "--host", "--cache-root", roots["cache"], "--rows-cap",
                str(LTM.ROWS_CAP)] + flags
        M1._ACTIVE[0] = fit
        LTM.LOADED.clear()
        try:
            with WQ.patch(M1, ROOTS=roots, GATE=GATE), \
                    WQ.patch(Z3, eval_carves=lambda: [(ds, carve) for ds in SETTINGS]), \
                    WQ.patch(Z3.RM, read=guard_read(Z3.RM.read, flags)):
                with M1.on_ml1() as W1:
                    W1.cache_root_ok(roots["cache"])
                    W1.gate_passed()
                    why = W1.wq_why(fit)
                    if why:
                        raise SystemExit(f"B1M: {why}")
                    with W1.on_wq() as W, LTM.on_arm(W), LTM.on_split(split, nk["fit_sha256"], out=str(LTM_OUT)):
                        if W.Z3P is not Z3P:
                            raise SystemExit("B1M: zwq's zp1c3p is not the one zrckpn was registered from")
                        rc = W.Z3P.read(argv)
                        nxt = LTM.next_block(split)
        finally:
            M1._ACTIVE[0] = None
            LTM.LOADED.clear()
    if rc:
        return rc
    miss = sorted({f"{ds}/{carve}" for ds in SETTINGS} - set(nxt["carves"]))
    if miss:
        raise SystemExit(f"B1M: {fit}'s read did not load the columns of {miss}")
    gone = [rel(p) for p in outs if not p.exists()]
    if gone:
        raise SystemExit(f"B1M: the read wrote no {gone}")
    write_json(fdir / "b1m_read.json", {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "arm": MLP, "variant": v,
                                        "carve": carve, "split": split, "reads": {rel(p): sha_file(p) for p in outs},
                                        "device": M1.device_of(env), "env": env, "gate_sha256": sha_file(ROOT / GATE),
                                        "next": nxt, "seconds": round(time.time() - t, 1),
                                        "zb1m_sha256": sha_src(__file__), "utc": utc()})
    log(f"read {fit} {v}: {time.time() - t:.0f} s")
    return 0


# ── the report ────────────────────────────────────────────────────────────────


def label(ci):
    """zg6r - zrckpn by its interval."""
    return GNN_ABOVE if ci[0] > 0 else MLP_ABOVE if ci[1] < 0 else LEVEL


def r100(x):
    return round(100 * float(x), 2)


def cols_record():
    """Each column record B1M's reads load: its path, sha256, identity check, rows and scorer."""
    out = {}
    for sp in SPLITS:
        for v in VARIANTS:
            for ds in SETTINGS:
                p = LTM_OUT / "cols" / sp / ds / CARVE[v] / "record.json"
                if not p.exists():
                    out[f"{sp} {ds}/{CARVE[v]}"] = None
                    continue
                r = read_json(p)
                out[f"{sp} {ds}/{CARVE[v]}"] = {
                    "record": rel(p), "sha256": sha_file(p), "identity": r.get("identity"), "rows": r.get("rows"),
                    "rows_with_an_anchor": r.get("rows_with_an_anchor"), "fit_sha256": r.get("fit_sha256"),
                    "training_carve": r.get("training_carve"), "seconds": r.get("seconds")}
    return out


def trains_ok(fit):
    """A fork's record names WQ's training carves of its split (the carves B1P checked the settings against)."""
    _sd, sp = FITS[fit]
    p = OUT / "fits" / "w" / fit / "screen.json"
    if not p.exists():
        return None
    return (read_json(p).get("wq") or {}).get("trains_on") == list(WQ.SPLITS[sp])


def report():
    import b1b as BB
    import b1f as B1F
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in IN_DOMAIN.items()}, "settings": {}, "missing": [],
           "trains_on_wq": {f: trains_ok(f) for f in FITS}}
    b1p_rep = read_json(B1P / "report.json") if (B1P / "report.json").exists() else {}
    for ds in SETTINGS:
        qs = read_json(BENCH / ds / "queries.json")
        gc = [len(q["golds"]) for q in qs]
        s = {"questions": len(qs), "overlap_with_training_b1p": ((b1p_rep.get("settings") or {}).get(ds) or {}).get(
                 "overlap_with_training"), "fits": {}, "in_domain": {}, "zero_shot": {}, "gnn_minus_mlp": {},
             "carry_over": {}, "zrckpn_minus_zrckp": {}}
        perq = {}
        for v in VARIANTS:
            for arm in ARMS:
                for sp in SPLITS:
                    for sd in SEEDS:
                        fit = fit_name(arm, sd, sp)
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
                    fits = [fit_name(arm, sd, sp) for sp in splits for sd in SEEDS]
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
                if (GNN, v, which) in means and (MLP, v, which) in means:
                    d = means[(GNN, v, which)] - means[(MLP, v, which)]
                    ci = BB.boot_ci(d)
                    pairs = [{"gnn": fit_name(GNN, sd, sp), "mlp": fit_name(MLP, sd, sp),
                              "R@5": round(float((perq[(GNN, v, fit_name(GNN, sd, sp))]["R@5"]
                                                  - perq[(MLP, v, fit_name(MLP, sd, sp))]["R@5"]).mean()), 4)}
                             for sp in splits for sd in SEEDS]
                    s["gnn_minus_mlp"][f"{which}/{CARVE[v]}"] = {"R@5": round(float(d.mean()), 4), "ci": ci,
                                                                 "label": label(ci), "per_fit_pair": pairs}
                if (MLP, v, which) in means and (OLD, v, which) in means:
                    d = means[(MLP, v, which)] - means[(OLD, v, which)]
                    s["zrckpn_minus_zrckp"][f"{which}/{CARVE[v]}"] = {"R@5": round(float(d.mean()), 4), "ci": BB.boot_ci(d)}
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
    rec["cols"] = cols_record()
    write_json(OUT / "cols.json", {"declared_in": DECLARED, "stage": STAGE, "cols": rec["cols"],
                                   "zb1m_sha256": sha_src(__file__), "utc": utc()})
    rec["zb1m_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    write_json(OUT / "report.json", rec)
    write_md(rec)
    log(f"report: {rel(OUT / 'report.md')}; missing {len(rec['missing'])}")
    return 0


def write_md(rec):
    def cell(c, key="R@5"):
        return f"{100 * c[key]:.1f}" if c else "--"

    def ci(c, key="R5_ci"):
        return f"{100 * c['R@5']:.1f} ({100 * c[key][0]:.1f}-{100 * c[key][1]:.1f})" if c else "--"

    def diff(c):
        return f"{100 * c['R@5']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f})" if c else "--"

    L = [f"# B1M report (declared in {DECLARED})", "",
         f"R@5 x100 of the {rec['candidate']} candidate on HippoRAG 2's settings (1,000 questions each), no LLM and no "
         "hyperlink. b1w = U2b's W2_S rule cut to the setting; b1t = the setting's text links only. zg6r and zrckp are "
         "B1P's reads; zrckpn is B1M's. In-domain = the mean over the fits that trained on the dataset (three seeds; six "
         "fits on 2wiki). 95% bootstrap intervals over the questions.", "",
         "## In-domain", "",
         "| setting | model | b1t | b1w (CI) | w - t (CI) | b1w FC@5 | b1w hit@1 | calls (any / peer) | gap (any / peer) |",
         "|---|---|---:|---:|---|---:|---:|---|---|"]
    for ds, s in rec["settings"].items():
        for arm in ARMS:
            cw, ct = s["in_domain"].get(f"{arm}/b1w"), s["in_domain"].get(f"{arm}/b1t")
            if cw is None and ct is None:
                continue
            cl, gp = s["calls"].get(f"{arm}/b1w"), s["gap_points"].get(f"{arm}/b1w")
            L.append(f"| {ds} | {arm} | {cell(ct)} | {ci(cw)} | {diff(s['carry_over'].get(f'{arm}/in_domain'))} | "
                     f"{cell(cw, 'FC@5')} | {cell(cw, 'hit@1')} | "
                     + (f"{cl['any']} / {cl['peer_reviewed']}" if cl else "--") + " | "
                     + (f"{gp['any']:+.1f} / {gp['peer_reviewed']:+.1f}" if gp else "--") + " |")
    L += ["", "Best published rows: " + "; ".join(
        f"{ds} {s['best_published']['any']['row']} {s['best_published']['any']['R@5']} / "
        f"{s['best_published']['peer_reviewed']['row']} {s['best_published']['peer_reviewed']['R@5']}"
        for ds, s in rec["settings"].items()), "",
          "## The two models of record: zg6r - zrckpn, paired by question", "",
          "| setting | read | carve | zg6r | zrckpn | zg6r - zrckpn (CI) | label | per fit pair (same seed and split) |",
          "|---|---|---|---:|---:|---|---|---|"]
    for ds, s in rec["settings"].items():
        for which in ("in_domain", "zero_shot"):
            for v in VARIANTS:
                g = s["gnn_minus_mlp"].get(f"{which}/{CARVE[v]}")
                if g is None:
                    continue
                L.append(f"| {ds} | {which.replace('_', '-')} | {CARVE[v]} | {cell(s[which].get(f'{GNN}/{CARVE[v]}'))} | "
                         f"{cell(s[which].get(f'{MLP}/{CARVE[v]}'))} | {diff(g)} | {g['label']} | "
                         + ", ".join(f"{100 * p['R@5']:+.1f}" for p in g["per_fit_pair"]) + " |")
    L += ["", "## Beside, deciding nothing", "",
          "| setting | read | carve | zrckpn - zrckp (CI) | zrckpn w - t (CI) |", "|---|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for which in ("in_domain", "zero_shot"):
            for v in VARIANTS:
                n = s["zrckpn_minus_zrckp"].get(f"{which}/{CARVE[v]}")
                if n is None:
                    continue
                L.append(f"| {ds} | {which.replace('_', '-')} | {CARVE[v]} | {diff(n)} | "
                         + (diff(s["carry_over"].get(f"{MLP}/{which}")) if v == "w" else "") + " |")
    L += ["", "Zero-shot reads (R@5 x100, b1w): " + "; ".join(
        f"{ds} {arm} {cell(s['zero_shot'].get(f'{arm}/b1w'))}" for ds, s in rec["settings"].items()
        for arm in ARMS if s["zero_shot"].get(f"{arm}/b1w")),
          "", "Training overlap (B1P's counts): " + "; ".join(
              f"{ds} {s['overlap_with_training_b1p']}" for ds, s in rec["settings"].items()),
          "", f"LTM's fits name WQ's training carves: {rec['trains_on_wq']}",
          "", f"Missing reads: {len(rec['missing'])}."]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ── self-test ─────────────────────────────────────────────────────────────────


def selftest():
    import zb1p as ZB
    import zltm as LTM
    import ltmcols as LTMC
    # 1. B1P's carves, settings, splits and in-domain map; LTM's fits
    assert (ZB.SETTINGS, ZB.VARIANTS, ZB.CARVE, ZB.SPLITS, ZB.SUF, ZB.IN_DOMAIN) == (
        SETTINGS, VARIANTS, CARVE, SPLITS, SUF, IN_DOMAIN), "B1P's carves and splits"
    assert ZB.read_flags("w")[1] == read_flags("w")[1] and read_flags("t")[3] == "outputs/b1m/fits/t", "read flags"
    assert {f: (MLP, *x) for f, x in FITS.items()} == dict(LTM.FITS) and LTM.ARM == MLP, "LTM's fits"
    assert ZB.FITS == tuple(fit_name(a, sd, sp) for a in (GNN, OLD) for sd in SEEDS for sp in SPLITS), "B1P's fits"
    for ds in SETTINGS:                                  # in-domain: the splits that train on the dataset, and only those
        assert set(IN_DOMAIN[ds]) == {sp for sp in SPLITS if ds in WQ.SPLITS[sp]}, f"in-domain {ds}"
    # 2. the columns' patches: the B1 carve named, LTM's folds read, both restored (also on a failure)
    seen = []
    orig_lf, orig_cc = LTMC.load_folds, LTMC.col_carves
    with WQ.patch(LTMC, load_folds=lambda out, split: seen.append((Path(out), split)) or "folds"):
        try:
            with on_b1_cols("musique", "b1w"):
                assert ("musique", "b1w") in LTMC.col_carves("WQ-hotpotqa"), "B1 carve named"
                assert set(orig_cc("WQ-hotpotqa")) < set(LTMC.col_carves("WQ-hotpotqa")), "the split's own carves kept"
                assert LTMC.load_folds(LTM_OUT, "WQ-hotpotqa") == "folds" and seen == [(Path(LTMC.OUT), "WQ-hotpotqa")]
                assert ("musique", "b1w") not in LTMC.train_carves("WQ-hotpotqa"), "no training carve"
                raise RuntimeError("inside")
        except RuntimeError:
            pass
    assert LTMC.load_folds is orig_lf and LTMC.col_carves is orig_cc, "patches restored"
    # 3. the read guard: flags added when absent, refused when other
    got = []
    g = guard_read(lambda a: got.append(a) or 0, read_flags("w"))
    assert g(["--name", "x"]) == 0 and got[-1][-4:] == read_flags("w"), "flags added"
    try:
        g(["--name", "x", "--read", "musique=b1w", "--out-root", "elsewhere"])
        raise AssertionError("other flags passed")
    except SystemExit:
        pass
    # 4. the fork skips only the source's own reads
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "src"
        (src / "reads").mkdir(parents=True)
        (src / "sub" / "reads").mkdir(parents=True)
        for f in ("models.pt", "read.json", "screen.json", "reads/a.npz", "sub/read.json", "sub/reads/b.npz"):
            (src / f).write_bytes(b"x")
        shutil.copytree(src, Path(t) / "dst", ignore=fork_ignore(src))
        kept = sorted(p.relative_to(Path(t) / "dst").as_posix() for p in (Path(t) / "dst").rglob("*") if p.is_file())
        assert kept == ["models.pt", "screen.json", "sub/read.json", "sub/reads/b.npz"], kept
    # 5. the label and the paths
    assert (label([0.001, 0.02]), label([-0.02, -0.001]), label([-0.01, 0.01]), label([0.0, 0.01])) == (
        GNN_ABOVE, MLP_ABOVE, LEVEL, LEVEL), "label"
    assert read_path(MLP, "scr-ltm-zrckpn-s0", "w", "2wiki") == OUT / "fits/w/scr-ltm-zrckpn-s0/reads/2wiki__b1w.npz"
    assert read_path(GNN, "scr-ml1-zg6r-s0-hp", "t", "musique") == B1P / "fits/t/scr-ml1-zg6r-s0-hp/reads/musique__b1t.npz"
    print("zb1m self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("lists", "cols", "fork", "read", "report", "selftest"))
    ap.add_argument("--dataset", choices=SETTINGS)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--name")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return selftest()
    need = {"lists": ("dataset", "variant"), "cols": ("split", "dataset", "variant"), "read": ("name", "variant")}
    for k in need.get(a.cmd, ()):
        if getattr(a, k) is None:
            raise SystemExit(f"{a.cmd}: --{k} is required")
    if a.cmd == "lists":
        return lists(a.dataset, a.variant)
    if a.cmd == "cols":
        return cols(a.split, a.dataset, a.variant, a.host)
    if a.cmd == "fork":
        return fork()
    if a.cmd == "read":
        return read(a.name, a.variant)
    if a.cmd == "report":
        return report()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
