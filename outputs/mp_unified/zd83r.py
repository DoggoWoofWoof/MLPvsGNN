"""D83's reads (docs/D83_UNIFIED_R_REFIT.md section 7), in the D83 world: the 18 D83 fits (zg6r, GNX, zrckpn; seeds 0, 1, 2;
both WQ splits) read on two arms, each read with D83's model changes for its kind (zd83m.model_changes: the label axis for
all; LAB beside NEXT in GNX's hook; LAB as a block after NEXT in zrckpn's pick), as the fit was trained.

    D83E   the six s1eval carves (outputs/abl/D83E: roots, gate, typed links, LAB); NEXT columns from the world's own
           outputs/ltm (zd83m.py ltm cols --arm D83E, from D83's fold fits)                          reported, not deciding
    B1V    b1v's three settings (outputs/b1v: roots, gate per setting, LAB; zd83b.py), NEXT columns outputs/b1v/ltm
                                                                                                     the deciding cells

The roots, gate and columns are named per read (zabl.arm_roots for D83E, zd83b's for b1v), so no read depends on what the
world's outputs/pko points at: the reads run in the world's train mode beside the fits still training (no mode switch).
GNX and zrckpn are read by zswp.py's engine (not edited; its OUT, arms, datasets and columns folder rebound here, as zswpr.py
does), zg6r by ML1's read (zabl.read's body). A fit is forked (models.pt checked, its D83 fit record named) before its reads.

    python outputs/host_ops/d83world.py run --mode train -- python outputs/mp_unified/zd83r.py fork --name FIT
    python outputs/host_ops/d83world.py run --mode train -- python outputs/mp_unified/zd83r.py read --arm A --name FIT --dataset D
    python outputs/mp_unified/zd83r.py selftest
"""
import os
import sys

# A zrckpn read runs in zltm's environment (6 BLAS / OpenMP / MKL threads, the GPU hidden) before numpy or torch loads; zswp
# sets it again after its imports and before the read (zswp's _MLP test: argv[1] is "read" and a zrckpn fit is named).
_MLP = sys.argv[1:2] == ["read"] and any(a.startswith("scr-ltm-zrckpn") for a in sys.argv)
_ENV0 = dict(os.environ)


def zrckpn_env():
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "6"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("NUMBA_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"):
        if _v in _ENV0:
            os.environ[_v] = _ENV0[_v]
        else:
            os.environ.pop(_v, None)


if _MLP:
    zrckpn_env()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zd83m as M83  # noqa: E402  (imports neither numpy nor torch at load; the label axis goes on before any model module)

ROOT = M83.ROOT
DECLARED = "docs/D83_UNIFIED_R_REFIT.md"
STAGE = "D83"
WORLD = "d83world"
SIX = ("squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
B1 = ("2wiki", "hotpotqa", "musique")
ARMS = {"D83E": ("s1eval", SIX), "B1V": ("b1v", B1)}
SEEDS = (0, 1, 2)
BASES = {"zg6r": "scr-ml1-zg6r", "gnx": "scr-gnx-nx", "zrckpn": "scr-ltm-zrckpn"}
FITS = tuple(f"{BASES[k]}-s{s}{suf}" for k in BASES for s in SEEDS for suf in ("", "-hp"))
SPLIT_OF = {"": "WQ-musique", "-hp": "WQ-hotpotqa"}
SCREEN_FITS = ROOT / "outputs" / "screen" / "fits"              # the world's own: D83's fits
FIT_RECORDS = ROOT / "outputs" / "d83" / "fits"                 # zd83t's per-fit D83 records
OUT = ROOT / "outputs" / "d83" / "reads"                        # zswp's OUT: forks and reads at OUT/fits/<arm>/<fit>
B1V_OUT = ROOT / "outputs" / "b1v"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zd83r: {msg}", flush=True)


sha_file, sha_src, rel, read_json, write_json = M83.sha_file, M83.sha_src, M83.rel, M83.read_json, M83.write_json


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fit_kind(fit):
    for k, base in BASES.items():
        for s in SEEDS:
            if fit in (f"{base}-s{s}", f"{base}-s{s}-hp"):
                return k
    raise SystemExit(f"D83R: {fit} is not one of D83's 18 read fits")


def fit_split(fit):
    fit_kind(fit)
    return SPLIT_OF["-hp" if fit.endswith("-hp") else ""]


def in_world():
    """The D83 world (D83WORLD=1, a tree named d83world), its fit, LTM, screen and d83 folders its own."""
    if os.environ.get("D83WORLD") != "1" or ROOT.name != WORLD:
        raise SystemExit("D83R: D83's reads run in the D83 world only (python outputs/host_ops/d83world.py run --mode train -- ...)")
    for d in ("screen", "ml1", "ltm", "gnx", "d83"):
        p = ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.join(os.path.realpath(ROOT), "outputs", d):
            raise SystemExit(f"D83R: {p} is not the D83 world's own folder")
    return True


def check(arm, ds=None):
    if arm not in ARMS:
        raise SystemExit(f"D83R: arm {arm}: one of {', '.join(ARMS)}")
    if ds is not None and ds not in ARMS[arm][1]:
        raise SystemExit(f"D83R: {arm} is read on {', '.join(ARMS[arm][1])}")
    return ARMS[arm][0]


def fork_dir(arm, fit):
    return OUT / "fits" / arm / fit


def roots_gate(M1, arm, ds):
    """ML1's roots, gate and typed links of the arm's carve of ds (relative paths, as ML1's)."""
    check(arm, ds)
    if arm == "D83E":
        import zabl as Z
        roots, gate = Z.arm_roots(M1, arm)
        return roots, gate, {"TYPED": f"outputs/abl/{arm}/typed"}
    import zd83b as B
    return {**M1.ROOTS, **B.read_roots()}, f"outputs/{B.CARVE}/gate_{ds}.json", {}


def cols_root(arm):
    """The NEXT columns of the arm's carves: D83E's in the world's own outputs/ltm (zd83m.py ltm cols writes there), b1v's in
    outputs/b1v/ltm (zd83b.py cols)."""
    import ltmcols as LTMC
    return Path(LTMC.OUT) if arm == "D83E" else B1V_OUT / "ltm"


def engine():
    """zswp's read engine with D83's names: the label axis set first (zswp imports zgnx and zml1s), its OUT D83's reads
    folder, the two arms and nine datasets added, the columns folder per arm. Returns zswp."""
    M83.labels_on()
    import zswp as S
    if getattr(S, "D83R", False):
        return S
    S.OUT, S.STAGE, S.DECLARED = OUT, STAGE, DECLARED
    S.ARMS = tuple(dict.fromkeys(tuple(S.ARMS) + tuple(ARMS)))
    S.DATASETS = tuple(dict.fromkeys(tuple(S.DATASETS) + SIX))
    S.cols_out = cols_root
    S.D83R = True
    return S


# ── fork ─────────────────────────────────────────────────────────────────────


def fork(fit):
    """The fit's folder (the world's outputs/screen/fits) copied without its reads to both arms' fork folders; models.pt
    checked; the fit's D83 record (zd83t: written after a fit that trained and stamped) must be there and name it."""
    in_world()
    kind = fit_kind(fit)
    src = SCREEN_FITS / fit
    frec = FIT_RECORDS / f"{fit}.json"
    if not (src / "models.pt").exists():
        raise SystemExit(f"D83R: {rel(src)} holds no models.pt")
    if not frec.exists() or read_json(frec).get("fit") != fit or read_json(frec).get("stage") != STAGE:
        raise SystemExit(f"D83R: {rel(frec)} is not D83's record of {fit} (zd83t.py train writes it)")
    want = sha_file(src / "models.pt")
    ignore = (lambda d, names: [x for x in names if Path(d) == src and x in ("reads", "read.json")])
    for arm in ARMS:
        dst = fork_dir(arm, fit)
        if (dst / "fork.json").exists():
            if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                raise SystemExit(f"D83R: {rel(dst)} holds other models than {rel(src)}")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp, ignore=ignore)
        if sha_file(tmp / "models.pt") != want:
            raise SystemExit(f"D83R: {rel(tmp)}'s models.pt is not the source's")
        os.replace(tmp, dst)
        write_json(dst / "fork.json", {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "fit": fit, "kind": kind,
                                       "source": rel(src), "models_sha256": want, "d83_fit_record": rel(frec),
                                       "d83_fit_record_sha256": sha_file(frec), "zd83r_sha256": sha_src(__file__), "utc": utc()})
        log(f"fork {fit} -> {rel(dst)}")
    return 0


# ── the reads ────────────────────────────────────────────────────────────────


def read_path(arm, fit, ds):
    return fork_dir(arm, fit) / "reads" / f"{ds}__{check(arm, ds)}.npz"


def marker_path(arm, fit, ds):
    return fork_dir(arm, fit) / f"d83_read_{ds}.json"


def read_zg6r(arm, fit, ds):
    """ML1's read of a zg6r fork on the arm's roots, gate and carve (zabl.read's body: zml1s's code, no stamp)."""
    S = engine()
    import zml1s as ML
    import zp1c3 as Z3
    import zwq as WQ
    carve = check(arm, ds)
    if ML.ALL_FITS.get(fit) is None or ML.ALL_FITS[fit][0] not in ML.BASE_OF:
        raise SystemExit(f"D83R: {fit} is not one of ML1's zg6r fits")
    env = ML.device_env()
    ML.where_ok(env, "read")
    _rec, k = ML.fit_stamp(fit)
    if ML.device_of(env) != k.get("device"):
        raise SystemExit(f"D83R: {fit} was fitted on {k.get('device')}; this read runs on {ML.device_of(env)}")
    roots, gate, typed = roots_gate(ML, arm, ds)
    flags = ["--read", f"{ds}={carve}", "--out-root", rel(OUT / "fits" / arm)]
    with WQ.patch(ML, ROOTS=roots, GATE=gate, **typed), WQ.patch(Z3, eval_carves=lambda: [(ds, carve)]), \
            WQ.patch(Z3.RM, read=S.A.guard_read(Z3.RM.read, flags)):
        rc, extra = ML.run_arm(fit, env, "read")
    return rc, {"device": ML.device_of(env), "gate": gate, "gate_sha256": sha_file(ROOT / gate), "roots": roots, "typed": typed,
                "extra": json.loads(json.dumps(extra, default=str))}


def read_pair(arm, fit, ds):
    """zswp.read (GNX under zgnx.on_gnx, zrckpn under zltm.on_ltm) with the arm's carve, roots, gate and typed links for the
    call: on b1v S.REC is the arm, so no typed links are asked of ABL's folder (zswpr.py's b1 reads)."""
    S = engine()
    carve = check(arm, ds)
    saved = (S.EVAL, S.roots_gate, S.REC)
    roots, gate, _typed = roots_gate(S.M1, arm, ds)
    S.EVAL, S.roots_gate = carve, (lambda _a: (roots, gate))
    if arm == "B1V":
        S.REC = arm
    try:
        rc = S.read(arm, fit, ds)
    finally:
        S.EVAL, S.roots_gate, S.REC = saved
    swp = fork_dir(arm, fit) / f"swp_read_{ds}.json"
    return rc, {"gate": gate, "gate_sha256": sha_file(ROOT / gate), "roots": roots,
                "engine_marker": rel(swp), "engine_marker_sha256": sha_file(swp) if swp.exists() else None}


def read(arm, fit, ds):
    """One D83 fit's read of the arm's carve of ds, with D83's model changes for the fit's kind; D83's marker after."""
    in_world()
    carve = check(arm, ds)
    kind = fit_kind(fit)
    fdir = fork_dir(arm, fit)
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"D83R: {rel(fdir)} is not forked (zd83r.py fork --name {fit})")
    out, marker = read_path(arm, fit, ds), marker_path(arm, fit, ds)
    if marker.exists() and out.exists():
        log(f"read {arm}/{fit}/{ds}: filed")
        return 0
    if kind == "zrckpn" and not _MLP:
        raise SystemExit("D83R: a zrckpn read starts with `zd83r.py read ... --name scr-ltm-zrckpn-*` (zltm's environment at start)")
    t = time.time()
    M83.labels_on()
    with M83.model_changes(kind):
        rc, info = (read_zg6r if kind == "zg6r" else read_pair)(arm, fit, ds)
        axis = M83.axis_record()
        lab = list(M83.LOADED)
    if rc:
        return rc
    if not out.exists():
        raise SystemExit(f"D83R: the read wrote no {rel(out)}")
    if kind != "zg6r" and not any(r.get("dataset") == ds and r.get("carve") == carve for r in lab):
        raise SystemExit(f"D83R: {fit}'s read of {ds}/{carve} loaded no LAB ({lab})")
    write_json(marker, {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "fit": fit, "kind": kind, "dataset": ds, "carve": carve,
                        "split": fit_split(fit), "read": rel(out), "read_sha256": sha_file(out), "axis": axis, "lab": lab,
                        "fork_sha256": sha_file(fdir / "fork.json"), **info, "seconds": round(time.time() - t, 1),
                        "zd83m_sha256": sha_src(M83.__file__), "zd83r_sha256": sha_src(__file__), "utc": utc()})
    log(f"read {arm}/{fit}/{ds}: {time.time() - t:.0f} s")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    assert len(FITS) == 18 and len(set(FITS)) == 18
    assert [fit_kind(f) for f in FITS].count("gnx") == 6 and fit_split("scr-gnx-nx-s1-hp") == "WQ-hotpotqa"
    assert fit_split("scr-ltm-zrckpn-s2") == "WQ-musique"
    for bad in ("scr-ml1-zspk-s0", "scr-ml1-zg6r-s3", "scr-ltm-zrckp-s0"):
        try:
            fit_kind(bad)
            raise AssertionError(f"{bad} must be refused")
        except SystemExit:
            pass
    keep = dict(os.environ)
    try:
        os.environ.pop("D83WORLD", None)
        for f in (lambda: fork(FITS[0]), lambda: read("B1V", FITS[0], "2wiki"), lambda: read("D83E", FITS[-1], "squad")):
            try:
                f()
                raise AssertionError("must refuse outside the D83 world")
            except SystemExit as e:
                assert "D83 world" in str(e), e
    finally:
        os.environ.clear()
        os.environ.update(keep)
    for a, d in (("B1V", "squad"), ("D83E", "b1v"), ("X", "2wiki")):
        try:
            check(a, d)
            raise AssertionError(f"{a}/{d} must be refused")
        except SystemExit:
            pass
    # in a fresh process (the label axis must go on before any model module): the engine's names, the roots, gates and typed
    # links per arm, the columns folders, the fits known to ML1, zgnx and zltm, and the model changes a read takes
    code = r"""
import sys, os
sys.path.insert(0, r'%s')
import zd83r as R, zd83m as M83
S = R.engine()
import zlink as ZL, zd83 as D, zml1s as ML, zgnx as GX, zltm as LTM, ltmcols as LTMC
assert ZL.FAMS == 5 and S.OUT == R.OUT and set(R.ARMS) <= set(S.ARMS) and set(R.SIX) <= set(S.DATASETS)
assert set(D.ARMS['D83E']) == set(R.SIX) and all(c == 's1eval' for c in D.ARMS['D83E'].values())
assert all((d, 's1eval') in LTMC.ALL_CARVES for d in R.SIX)
for f in R.FITS:
    k = R.fit_kind(f)
    assert (f in ML.ALL_FITS and ML.ALL_FITS[f][0] in ML.BASE_OF) if k == 'zg6r' else (f in S.GNX_FITS if k == 'gnx' else f in S.MLP_FITS)
    assert R.fit_split(f) == (ML.ALL_FITS[f][2] if k == 'zg6r' else S.fit_split(f)), f
r, g, t = R.roots_gate(ML, 'D83E', 'metaqa')
assert r['cache'] == 'outputs/abl/D83E/cache' and r['links'] == 'outputs/abl/D83E/links' and g == 'outputs/abl/D83E/gate.json'
assert t == {'TYPED': 'outputs/abl/D83E/typed'}
r, g, t = R.roots_gate(ML, 'B1V', 'musique')
assert r['cache'] == 'outputs/b1v/cache' and r['pools'] == 'outputs/b1v/pools' and g == 'outputs/b1v/gate_musique.json' and t == {}
assert R.cols_root('D83E') == __import__('pathlib').Path(LTMC.OUT) and R.cols_root('B1V') == R.B1V_OUT / 'ltm'
assert S.cols_out('B1V') == R.B1V_OUT / 'ltm' and S.read_path('B1V', R.FITS[0], 'x').parent.parent == R.fork_dir('B1V', R.FITS[0])
with M83.model_changes('gnx'):
    assert GX.IN == M83.GNX_IN == 35 and GX.D83_LAB
with M83.model_changes('zrckpn'):
    assert LTM.D83_LAB and tuple(LTM.PICK[-2:]) == (LTM.NEXT, M83.LAB) and not LTM._ARM
print('engine OK')
""" % str(HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0 and "engine OK" in r.stdout, (r.stdout[-1500:], r.stderr[-3000:])
    # a zrckpn read's process starts in zltm's environment
    code = (r"import sys; sys.argv = ['zd83r.py', 'read', '--name', 'scr-ltm-zrckpn-s0']; sys.path.insert(0, r'%s'); import os; "
            r"import zd83r; assert os.environ['OMP_NUM_THREADS'] == os.environ['OPENBLAS_NUM_THREADS'] == '6' and "
            r"os.environ['CUDA_VISIBLE_DEVICES'] == '' and zd83r._MLP; print('env OK')" % str(HERE))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0 and "env OK" in r.stdout, r.stderr[-2000:]
    print("zd83r self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zd83r.py")
    ap.add_argument("cmd", choices=("selftest", "fork", "read"))
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--name")
    ap.add_argument("--dataset")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "fork":
        if not a.name:
            raise SystemExit("D83R fork: --name")
        return fork(a.name)
    if not (a.arm and a.name and a.dataset):
        raise SystemExit("D83R read: --arm, --name and --dataset")
    return read(a.arm, a.name, a.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
