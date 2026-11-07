"""SWP part B, the same-input pair (GNX, zrckpn) on the swap's carves of SQuAD, MetaQA and WebQSP (docs/SWP_S_TO_R_SWAP.md section 4).

The swap S -> R with N and K in, on the datasets whose record has no small b1 carve. A cell is a dataset's full dev carve (s1eval, the
carve ABL and ML1 read) built with one thing in the structural slot, the ner and knn stores in:
    S   the dataset's own structural graph   SQuAD: title mentions (arm SNK)    MetaQA, WebQSP: the KB triples (the record, REC)
    R   the rule graph (W2_S)                SQuAD: the record (REC)             MetaQA, WebQSP: arm RK
    -   the slot empty (arm NK)
The arms are zabl.py's (SNK and NK are ARM_SWP, RK is a filed ABL arm); REC is the record's own roots. Nothing trains; zabl.py, zidpa.py,
zidpb.py, zb1h.py, zgnx.py, zltm.py and ltmcols.py are imported and edited by nothing here. Everything written is under outputs/swp/.

    python outputs/mp_unified/zswp.py selftest
    python outputs/mp_unified/zswp.py fork --arm REC|RK|SNK|NK              # GNX's six and zrckpn's six fits, without their reads
    python outputs/mp_unified/zswp.py lists --arm A --dataset D --host      # LTM's candidate lists of the arm's carve (the host)
    python outputs/mp_unified/zswp.py cols --arm A --dataset D --split S --host   # the next-hop columns of LTM's fold fits on that carve
    python outputs/mp_unified/zswp.py read --arm A --name FIT --dataset D   # one fit's read of the arm's carve (the host's GPU / CPU)
    python outputs/mp_unified/zswp.py report                                # the laptop: outputs/swp/report.{json,md}
"""
import os
import sys

# A zrckpn read runs in zltm's environment (DOB's): 6 BLAS / OpenMP / MKL threads, the GPU hidden, before numpy or torch loads
# (zb1h.py does this for read-mlp). zabl.py and zb1h.py put the PKO processes' 2 threads (and NUMBA_NUM_THREADS, CUBLAS_WORKSPACE_CONFIG)
# into the environment at import, so the block runs again after the imports and again before the read; the variables the
# queue gave the process come back. A GNX read and every lists / cols / report call keep the PKO environment.
_MLP = sys.argv[1:2] == ["read"] and any(a.startswith("scr-ltm-zrckpn") for a in sys.argv)
_ENV0 = dict(os.environ)


def zrckpn_env():
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "6"                                  # zltm's THREADS
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                   # a CPU read never sees the GPU (zltm's)
    for _v in ("NUMBA_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"):
        if _v in _ENV0:
            os.environ[_v] = _ENV0[_v]
        else:
            os.environ.pop(_v, None)


if _MLP:
    zrckpn_env()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zabl as Z  # noqa: E402  (ABL: arms, roots, gates, per-question cells; read only)
import zidpa as A  # noqa: E402  (IDP A1: GNX's fits and the read's guard; read only)
import zb1h as H  # noqa: E402  (B1H: zrckpn's fits, fork rules and the read's shape; read only)

if _MLP:
    zrckpn_env()                                               # zabl, zidpa and zb1h have just set the PKO environment

GX, M1 = A.GX, A.M1
WQ = M1.WQ
DECLARED = "docs/SWP_S_TO_R_SWAP.md"
STAGE = "SWP-B"
OUT = ROOT / "outputs" / "swp"
EVAL = Z.CARVE                                             # "s1eval"
REC = Z.CHECK_ARM                                          # "REC": the record's own roots and gate
ARMS = (REC, "RK", "SNK", "NK")
DATASETS = ("squad", "metaqa", "webqsp")
# the arm that holds each slot in each dataset (REC is the record: R on SQuAD, S on the KB datasets)
SLOT_ARM = {"squad": {"S": "SNK", "R": REC, "-": "NK"}, "metaqa": {"S": REC, "R": "RK", "-": "NK"},
            "webqsp": {"S": REC, "R": "RK", "-": "NK"}}
GNX, MLP = "gnx", H.MLP
MODELS = (GNX, MLP)
SEEDS, SPLITS = A.SEEDS, A.SPLITS
GNX_FITS = tuple(GX.FITS)
MLP_FITS = tuple(H.FITS[MLP])
ALL_FITS = GNX_FITS + MLP_FITS
assert len(GNX_FITS) == 6 and len(MLP_FITS) == 6 and OUT != A.OUT


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zswp: {msg}", flush=True)


utc, sha_file, sha_src, rel, read_json, write_json = A.utc, A.sha_file, A.sha_src, A.rel, A.read_json, A.write_json


def check_arm(arm):
    if arm not in ARMS:
        raise SystemExit(f"SWP: arm {arm}: one of {', '.join(ARMS)}")


def arm_dir(arm):
    """The arm's roots (pools, look, cache, links, kt, gate...): ABL's folder; REC is the record's."""
    return Z.arm_dir(arm) if arm != REC else None


def cols_out(arm):
    """Where the arm's next-hop columns and lists live; REC reads LTM's own (outputs/ltm, read only)."""
    import ltmcols as LTMC
    return Path(LTMC.OUT) if arm == REC else OUT / "ltm" / arm


def fit_split(fit):
    """GNX's and zrckpn's fits: the WQ split a fit trained on."""
    if fit in GNX_FITS:
        return GX.FITS[fit][2]
    import zltm as LTM
    return LTM.FITS[fit][2]


def fit_model(fit):
    return GNX if fit in GNX_FITS else MLP


def fits_of(model, splits):
    names = GNX_FITS if model == GNX else MLP_FITS
    return [f for f in names if fit_split(f) in splits]


def in_domain_splits(ds):
    return Z.in_domain_splits(ds)


# ── fork ─────────────────────────────────────────────────────────────────────


def fork(arm):
    """GNX's six and zrckpn's six fits copied without their reads to outputs/swp/fits/<arm>/<fit>; models.pt checked."""
    check_arm(arm)
    for fit in ALL_FITS:
        src, dst = A.SCREEN_FITS / fit, OUT / "fits" / arm / fit
        if not (src / "models.pt").exists():
            raise SystemExit(f"SWP: {rel(src)} holds no models.pt")
        want = sha_file(src / "models.pt")
        if (dst / "fork.json").exists():
            if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                raise SystemExit(f"SWP: {rel(dst)} holds other models than {rel(src)}")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp, ignore=H.fork_ignore(src))
        if sha_file(tmp / "models.pt") != want:
            raise SystemExit(f"SWP: {rel(tmp)}'s models.pt is not the source's")
        os.replace(tmp, dst)
        write_json(dst / "fork.json", {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "fit": fit, "source": rel(src),
                                       "models_sha256": want, "zswp_sha256": sha_src(__file__), "utc": utc()})
        log(f"fork {arm}: {rel(src)} -> {rel(dst)}")
    return 0


# ── LTM's lists and columns on an arm's carve ────────────────────────────────


@contextlib.contextmanager
def on_cols():
    """ltmcols with load_folds reading LTM's own fold fits wherever the columns are written (the carve is an eval carve of the
    split already: col_carves names it)."""
    import ltmcols as LTMC
    orig_lf = LTMC.load_folds
    with WQ.patch(LTMC, load_folds=lambda _out, split: orig_lf(LTMC.OUT, split)):
        yield LTMC


def lists(arm, ds, host=False):
    check_arm(arm)
    if arm == REC:
        raise SystemExit("SWP: REC reads LTM's own columns; nothing to build")
    import ltmcols as LTMC
    if host:
        import lean_host as LH
        LH.substitute()
    LTMC.lists(ds, EVAL, root=Z.arm_dir(arm), out=cols_out(arm))
    return 0


def cols(arm, split, ds, host=False):
    check_arm(arm)
    if arm == REC:
        raise SystemExit("SWP: REC reads LTM's own columns; nothing to build")
    import ltmcols as LTMC
    if host:
        import lean_host as LH
        LH.substitute()
    with on_cols():
        LTMC.cols(split, ds, EVAL, root=Z.arm_dir(arm), out=cols_out(arm), host=host)
    return 0


# ── the reads ────────────────────────────────────────────────────────────────


def read_flags(arm, ds):
    return ["--read", f"{ds}={EVAL}", "--out-root", rel(OUT / "fits" / arm)]


def read_path(arm, fit, ds):
    return OUT / "fits" / arm / fit / "reads" / f"{ds}__{EVAL}.npz"


def roots_gate(arm):
    """ML1's roots and gate for the arm (zabl.arm_roots); the arm's typed links for the chained carves."""
    roots, gate = Z.arm_roots(M1, arm)
    return roots, gate


def read(arm, fit, ds):
    """One fit's read of the arm's carve: GNX under zgnx.on_gnx (zidpb.read's body), zrckpn under zltm.on_ltm (zb1h.read_mlp's
    body), both on the arm's roots and gate, with the arm's columns (REC: LTM's own)."""
    import zp1c3 as Z3
    check_arm(arm)
    if ds not in DATASETS:
        raise SystemExit(f"SWP: --dataset is one of {', '.join(DATASETS)}")
    if fit not in ALL_FITS:
        raise SystemExit(f"SWP: {fit} is not one of GNX's or zrckpn's fits")
    split = fit_split(fit)
    fdir = OUT / "fits" / arm / fit
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"SWP: {rel(fdir)} is not forked (zswp.py fork --arm {arm})")
    out = read_path(arm, fit, ds)
    marker = fdir / f"swp_read_{ds}.json"
    if marker.exists() and out.exists():
        log(f"read {arm}/{fit}/{ds}: filed")
        return 0
    roots, gate = roots_gate(arm)
    flags = read_flags(arm, ds)
    colroot = cols_out(arm)
    typed = {} if arm == REC else ({"TYPED": f"outputs/abl/{arm}/typed"} if hasattr(M1, "TYPED") else {})
    t = time.time()
    rc, extra, nxt = 1, {}, None
    if fit in GNX_FITS:
        G = GX.import_g()
        call = GX.DEP.iso_call()
        with GX.on_gnx(G, call):
            env = M1.device_env()
            M1.where_ok(env, "read")
            _rec, k = M1.fit_stamp(fit)
            if M1.device_of(env) != k.get("device"):
                raise SystemExit(f"SWP: {fit} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
            nk = k.get("next") or {}
            if (nk.get("split"), nk.get("columns")) != (split, list(GX.LTMC.COLS)) or not nk.get("fit_sha256"):
                raise SystemExit(f"SWP: {fit} was not trained on LTM's columns of {split} ({nk})")
            GX.LOADED.clear()
            try:
                with WQ.patch(M1, ROOTS=roots, GATE=gate, **typed), \
                        WQ.patch(Z3, eval_carves=lambda: [(ds, EVAL)]), \
                        WQ.patch(Z3.RM, read=A.guard_read(Z3.RM.read, flags)), \
                        GX.on_split(split, nk["fit_sha256"], out=str(colroot)):
                    rc, extra = M1.run_arm(fit, env, "read")
                    nxt = GX.next_block(split)
            finally:
                GX.LOADED.clear()
    else:
        if not _MLP:
            raise SystemExit("SWP: a zrckpn read starts with `zswp.py read ... --name scr-ltm-zrckpn-*` (zltm's environment is set at start)")
        import torch
        zrckpn_env()
        torch.set_num_threads(6)                          # zltm's THREADS (torch may have loaded under the PKO environment)
        import zltm as LTM
        import ltmcols as LTMC
        Z3P = LTM.import_arm()
        with LTM.on_ltm():
            env = M1.device_env()
            M1.where_ok(env, "read")
            _rec, k = M1.fit_stamp(fit)
            if M1.device_of(env) != k.get("device"):
                raise SystemExit(f"SWP: {fit} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
            nk = k.get("next") or {}
            if (nk.get("declared_in"), nk.get("split"), nk.get("columns")) != (LTM.DECLARED, split, list(LTMC.COLS)) or \
                    not nk.get("fit_sha256"):
                raise SystemExit(f"SWP: {fit} was not trained on LTM's columns of {split} ({nk})")
            argv = ["--name", fit, "--device", "cpu", "--host", "--cache-root", roots["cache"], "--rows-cap",
                    str(LTM.ROWS_CAP)] + flags
            M1._ACTIVE[0] = fit
            LTM.LOADED.clear()
            try:
                with WQ.patch(M1, ROOTS=roots, GATE=gate, **typed), \
                        WQ.patch(Z3, eval_carves=lambda: [(ds, EVAL)]), \
                        WQ.patch(Z3.RM, read=A.guard_read(Z3.RM.read, flags)):
                    with M1.on_ml1() as W1:
                        W1.cache_root_ok(roots["cache"])
                        W1.gate_passed()
                        why = W1.wq_why(fit)
                        if why:
                            raise SystemExit(f"SWP: {why}")
                        with W1.on_wq() as W, LTM.on_arm(W), LTM.on_split(split, nk["fit_sha256"], out=str(colroot)):
                            if W.Z3P is not Z3P:
                                raise SystemExit("SWP: zwq's zp1c3p is not the one zrckpn was registered from")
                            rc = W.Z3P.read(argv)
                            nxt = LTM.next_block(split)
            finally:
                M1._ACTIVE[0] = None
                LTM.LOADED.clear()
    if rc:
        return rc
    if f"{ds}/{EVAL}" not in nxt["carves"]:
        raise SystemExit(f"SWP: {fit}'s read did not load the columns of {ds}/{EVAL}")
    if nxt["fit_sha256"] != nk["fit_sha256"]:
        raise SystemExit(f"SWP: {fit}'s read loaded columns of the scorer {str(nxt['fit_sha256'])[:12]}, not its own {nk['fit_sha256'][:12]}")
    if not out.exists():
        raise SystemExit(f"SWP: the read wrote no {rel(out)}")
    write_json(marker, {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "fit": fit, "model": fit_model(fit), "dataset": ds,
                        "carve": EVAL, "split": split, "read": rel(out), "read_sha256": sha_file(out), "device": M1.device_of(env),
                        "env": env, "gate": gate, "gate_sha256": sha_file(ROOT / gate), "columns_root": rel(colroot), "next": nxt,
                        "extra": json.loads(json.dumps(extra, default=str)), "seconds": round(time.time() - t, 1),
                        "zswp_sha256": sha_src(__file__), "utc": utc()})
    log(f"read {arm}/{fit}/{ds}: {time.time() - t:.0f} s")
    return 0


# ── the report ───────────────────────────────────────────────────────────────


def cell(arm, ds, model, regime):
    """Per-question R@5 of the model's fits in the regime (in-domain or held out), from the arm's forks."""
    import p3c_claim1 as P3
    ind = in_domain_splits(ds)
    splits = [sp for sp in SPLITS if (sp in ind) == (regime == "in-domain")]
    fits = fits_of(model, splits)
    got = []
    for f in fits:
        fd = OUT / "fits" / arm / f
        if not read_path(arm, f, ds).exists():
            return None
        r = P3.load_read(ROOT, rel(fd), ds)
        got.append((r["ids"], r["gt"], r["m"][:, 0], r["sha256"]))
    if not fits:
        return None
    if any(not np.array_equal(g[0], got[0][0]) for g in got):
        raise SystemExit(f"SWP: {arm}/{ds}: the fits' questions differ")
    return {"ids": got[0][0], "gt": got[0][1], "per_fit": [g[2] for g in got], "sha256": [g[3] for g in got], "fits": fits}


def point(c):
    ok = c["gt"] > 0
    return float(np.mean(np.stack(c["per_fit"]), 0)[ok].mean())


def report():
    rec = {"declared_in": DECLARED, "stage": STAGE, "carve": EVAL, "boot": Z.BOOT, "slot_arm": SLOT_ARM, "datasets": {}, "missing": []}
    for ds in DATASETS:
        d = {"cells": {}, "swap": {}, "gap": {}}
        regimes = ("in-domain", "held-out") if in_domain_splits(ds) != tuple(SPLITS) else ("in-domain",)
        for regime in ("in-domain", "held-out"):
            if not [sp for sp in SPLITS if (sp in in_domain_splits(ds)) == (regime == "in-domain")]:
                continue
            cells = {}
            for slot, arm in SLOT_ARM[ds].items():
                for model in MODELS:
                    c = cell(arm, ds, model, regime)
                    if c is None:
                        rec["missing"].append(f"{regime}/{ds}/{arm}/{model}")
                        continue
                    cells[(slot, model)] = c
                    d["cells"][f"{regime}/{slot}/{model}"] = {"arm": arm, "R@5": round(point(c), 4), "fits": len(c["per_fit"]),
                                                              "per_fit": [round(float(np.mean(x[c['gt'] > 0])), 4) for x in c["per_fit"]]}
            for model in MODELS:
                for a_, b_ in (("S", "R"), ("-", "R"), ("-", "S")):
                    if (a_, model) in cells and (b_, model) in cells:
                        d["swap"][f"{regime}/{model}/{b_} - {a_}"] = Z.compare(cells[(a_, model)], cells[(b_, model)])
            for slot in SLOT_ARM[ds]:
                if (slot, GNX) in cells and (slot, MLP) in cells:
                    d["gap"][f"{regime}/{slot}"] = Z.compare(cells[(slot, MLP)], cells[(slot, GNX)])
        rec["datasets"][ds] = d
    rec["zswp_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "report.json", rec)
    write_md(rec)
    log(f"report: {rel(OUT / 'report.md')}; missing {len(rec['missing'])}")
    return 0


def _p(c):
    return "--" if c is None else f"{100 * c['R@5']:.1f}"


def _d(c):
    return "--" if c is None else f"{100 * c['diff']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}) {c['call']}"


def write_md(rec):
    L = ["# SWP part B: the same-input pair on the swap S -> R (N and K in), SQuAD, MetaQA, WebQSP", "",
         f"Declared in {DECLARED}. R@5 in points on each dataset's full dev carve (s1eval), 3-seed means over the splits of the regime, "
         f"paired 95% intervals over the questions ({rec['boot']} resamples). Slot S: the dataset's own structural graph (SQuAD: title "
         f"mentions; MetaQA, WebQSP: the KB triples). Slot R: the rule graph. Slot -: empty. The ner and knn stores are in every cell. "
         f"Fits are the models of record (nothing trains): on the KB datasets the record's slot is S, on SQuAD it is R, so the other "
         f"slot is a swap at read time in the direction named in the row. No threshold, no adoption call.", "",
         "| dataset | regime | model | S | R | - | R - S | R - (none) | S - (none) |", "|---|---|---|---:|---:|---:|---|---|---|"]
    for ds, d in rec["datasets"].items():
        for regime in ("in-domain", "held-out"):
            for model in MODELS:
                if f"{regime}/S/{model}" not in d["cells"] and f"{regime}/R/{model}" not in d["cells"]:
                    continue
                sw = d["swap"]
                L.append(f"| {ds} | {regime} | {model} | " + " | ".join(_p(d["cells"].get(f"{regime}/{s}/{model}")) for s in ("S", "R", "-"))
                         + f" | {_d(sw.get(f'{regime}/{model}/R - S'))} | {_d(sw.get(f'{regime}/{model}/R - -'))} | "
                         + f"{_d(sw.get(f'{regime}/{model}/S - -'))} |")
    L += ["", "## Does message passing matter more with structure in? GNX minus zrckpn, same inputs", "",
          "| dataset | regime | S | R | - |", "|---|---|---|---|---|"]
    for ds, d in rec["datasets"].items():
        for regime in ("in-domain", "held-out"):
            if not any(k.startswith(regime) for k in d["gap"]):
                continue
            L.append(f"| {ds} | {regime} | " + " | ".join(_d(d["gap"].get(f"{regime}/{s}")) for s in ("S", "R", "-")) + " |")
    if rec["missing"]:
        L += ["", f"Missing cells: {len(rec['missing'])}"] + [f"- {m}" for m in rec["missing"][:30]]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    # 1. the mapping of slots to arms is the swap: each dataset's three slots are three different cells, the record is REC once
    for ds, m in SLOT_ARM.items():
        assert set(m) == {"S", "R", "-"} and len(set(m.values())) == 3, (ds, m)
        assert list(m.values()).count(REC) == 1 and all(a in ARMS for a in m.values())
    assert SLOT_ARM["squad"]["R"] == REC and SLOT_ARM["metaqa"]["S"] == REC == SLOT_ARM["webqsp"]["S"]
    assert Z.ARM_ALL["SNK"][0] == "S" and Z.ARM_ALL["NK"][0] == "" and Z.ARM_ALL["RK"][0] == "R"
    assert all(Z.ARM_ALL[a][1:] == (1, 1) for a in ("SNK", "NK", "RK")), "N and K are in every cell"
    # 2. the fits: GNX's six and zrckpn's six, two splits, three seeds, in-domain splits per dataset
    assert len(ALL_FITS) == 12 and len(set(ALL_FITS)) == 12
    assert [fit_split(f) for f in GNX_FITS].count("WQ-hotpotqa") == 3 and [fit_split(f) for f in MLP_FITS].count("WQ-musique") == 3
    assert fits_of(GNX, ("WQ-hotpotqa",)) == [f for f in GNX_FITS if fit_split(f) == "WQ-hotpotqa"] and len(fits_of(MLP, SPLITS)) == 6
    assert in_domain_splits("metaqa") == () and set(in_domain_splits("squad")) == set(SPLITS) == set(in_domain_splits("webqsp"))
    # 3. paths and flags
    assert read_path("RK", "scr-gnx-nx-s0", "metaqa") == OUT / "fits/RK/scr-gnx-nx-s0/reads/metaqa__s1eval.npz"
    assert read_flags("SNK", "squad") == ["--read", "squad=s1eval", "--out-root", "outputs/swp/fits/SNK"], read_flags("SNK", "squad")
    assert cols_out("NK") == OUT / "ltm" / "NK" and cols_out(REC).name == "ltm"
    assert arm_dir(REC) is None and arm_dir("RK") == Z.arm_dir("RK")
    # 4. an arm outside the swap is refused
    for bad in ("SRK", "K"):
        try:
            check_arm(bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    # 5. the fork keeps models.pt and drops only the source's reads (zb1h's rule)
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "src"
        (src / "reads").mkdir(parents=True)
        for f in ("models.pt", "read.json", "screen.json", "reads/a.npz"):
            (src / f).write_bytes(b"x")
        shutil.copytree(src, Path(t) / "dst", ignore=H.fork_ignore(src))
        kept = sorted(p.relative_to(Path(t) / "dst").as_posix() for p in (Path(t) / "dst").rglob("*") if p.is_file())
        assert kept == ["models.pt", "screen.json"], kept
    # 6. zrckpn's environment: 6 threads, the GPU hidden, zabl's PKO-only variables gone unless the queue gave them
    saved = dict(os.environ)
    try:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = "2"
        os.environ["NUMBA_NUM_THREADS"] = "2"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        zrckpn_env()
        assert all(os.environ[v] == "6" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert ("NUMBA_NUM_THREADS" in os.environ) == ("NUMBA_NUM_THREADS" in _ENV0)
        assert ("CUBLAS_WORKSPACE_CONFIG" in os.environ) == ("CUBLAS_WORKSPACE_CONFIG" in _ENV0)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    # 7. the comparison on synthetic cells (ABL's paired bootstrap)
    ids = np.arange(200)
    a = {"ids": ids, "gt": np.ones(200), "per_fit": [np.full(200, 0.5), np.full(200, 0.5)]}
    b = {"ids": ids, "gt": np.ones(200), "per_fit": [np.full(200, 0.8), np.full(200, 0.8)]}
    c = Z.compare(a, b)
    assert c["call"] == "ABOVE" and abs(c["diff"] - 0.3) < 1e-9 and point(a) == 0.5
    print("zswp self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fork", "lists", "cols", "read", "report", "selftest"))
    ap.add_argument("--arm")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--split")
    ap.add_argument("--name")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return selftest()
    if a.cmd in ("fork", "lists", "cols", "read") and a.arm is None:
        raise SystemExit(f"{a.cmd}: --arm is required")
    if a.cmd == "fork":
        return fork(a.arm)
    if a.cmd == "lists":
        if a.dataset is None:
            raise SystemExit("lists: --dataset is required")
        return lists(a.arm, a.dataset, a.host)
    if a.cmd == "cols":
        if a.dataset is None or a.split is None:
            raise SystemExit("cols: --dataset and --split are required")
        return cols(a.arm, a.split, a.dataset, a.host)
    if a.cmd == "read":
        if a.dataset is None or a.name is None:
            raise SystemExit("read: --dataset and --name are required")
        return read(a.arm, a.name, a.dataset)
    if a.cmd == "report":
        return report()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
