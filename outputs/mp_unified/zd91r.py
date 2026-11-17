"""D91's reads (docs/D91_D83_WITHOUT_LABELS.md section 6), in the D91 world: the 18 D91 fits (zg6r, GNX, zrckpn; seeds 0, 1, 2;
both WQ splits) read on two arms with the record's model code (three families, no label axis, no LAB), as they were trained.

    D91E   the six s1eval carves (outputs/abl/D91E: roots, gate, typed links); NEXT columns from the world's own outputs/ltm
           (zd91m.py ltm cols --arm D91E, from D91's fold fits)                                   reported, not deciding
    D91B   B1's three settings (outputs/abl/D91B: roots, one gate over the three; zd91.py), NEXT columns
           outputs/abl/D91B/ltm (zd91m.py b1 cols)                                                the deciding cells

zd83r.py's reads with D91's names and without D83's model changes. The roots, gate and columns are named per read, so no read
depends on what the world's outputs/pko points at: the reads run in the world's train mode beside the fits still training.
GNX and zrckpn are read by zswp.py's engine (not edited; its OUT, arms, datasets and columns folder rebound here), zg6r by
ML1's read (zabl.read's body). A fit is forked (models.pt checked against its D91 fit record) before its reads.

    python outputs/host_ops/d91world.py run --mode train -- python outputs/mp_unified/zd91r.py fork --name FIT
    python outputs/host_ops/d91world.py run --mode train -- python outputs/mp_unified/zd91r.py read --arm A --name FIT --dataset D
    python outputs/mp_unified/zd91r.py selftest
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
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DECLARED = "docs/D91_D83_WITHOUT_LABELS.md"
STAGE = "D91"
WORLD = "d91world"
OWN = ("screen", "ml1", "ltm", "gnx", "d91")
SIX = ("squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
B1 = ("2wiki", "hotpotqa", "musique")
ARMS = {"D91E": ("s1eval", SIX), "D91B": ("b1v", B1)}       # D91B keeps b1v's carve name (zd91.CARVES)
SEEDS = (0, 1, 2)
BASES = {"zg6r": "scr-ml1-zg6r", "gnx": "scr-gnx-nx", "zrckpn": "scr-ltm-zrckpn"}
FITS = tuple(f"{BASES[k]}-s{s}{suf}" for k in BASES for s in SEEDS for suf in ("", "-hp"))
SPLIT_OF = {"": "WQ-musique", "-hp": "WQ-hotpotqa"}
SCREEN_FITS = ROOT / "outputs" / "screen" / "fits"              # the world's own: D91's fits
FIT_RECORDS = ROOT / "outputs" / "d91" / "fits"                 # zd91t's per-fit D91 records
OUT = ROOT / "outputs" / "d91" / "reads"                        # zswp's OUT: forks and reads at OUT/fits/<arm>/<fit>
B1_ARM = "D91B"                                                 # B1's three settings: passages, no typed links


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zd91r: {msg}", flush=True)


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
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fit_kind(fit):
    for k, base in BASES.items():
        for s in SEEDS:
            if fit in (f"{base}-s{s}", f"{base}-s{s}-hp"):
                return k
    raise SystemExit(f"D91R: {fit} is not one of D91's 18 read fits")


def fit_split(fit):
    fit_kind(fit)
    return SPLIT_OF["-hp" if fit.endswith("-hp") else ""]


def in_world():
    """The D91 world (D91WORLD=1, a tree named d91world), its fit, LTM, screen and d91 folders its own."""
    if os.environ.get("D91WORLD") != "1" or ROOT.name != WORLD:
        raise SystemExit("D91R: D91's reads run in the D91 world only (python outputs/host_ops/d91world.py run --mode train -- ...)")
    for d in OWN:
        p = ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.join(os.path.realpath(ROOT), "outputs", d):
            raise SystemExit(f"D91R: {p} is not the D91 world's own folder")
    return True


def record_axis():
    """The record's three families in force (no D83 label axis, no D83 module). Returns zlink."""
    import zlink as ZL
    if getattr(ZL, "D83_LABELS", False) or ZL.FAMS != 3 or ZL.SETTINGS.get("families") != 3 or "labels" in ZL.SETTINGS:
        raise SystemExit(f"D91R: zlink holds {ZL.FAMS} families; D91 reads on the record's three")
    if "zd83m" in sys.modules:
        raise SystemExit("D91R: zd83m (D83's model changes) is imported in a D91 process")
    return ZL


def check(arm, ds=None):
    if arm not in ARMS:
        raise SystemExit(f"D91R: arm {arm}: one of {', '.join(ARMS)}")
    if ds is not None and ds not in ARMS[arm][1]:
        raise SystemExit(f"D91R: {arm} is read on {', '.join(ARMS[arm][1])}")
    return ARMS[arm][0]


def fork_dir(arm, fit):
    return OUT / "fits" / arm / fit


def roots_gate(M1, arm, ds):
    """ML1's roots, gate and typed links of the arm's carve of ds (relative paths, as ML1's)."""
    check(arm, ds)
    import zabl as Z
    roots, gate = Z.arm_roots(M1, arm)                            # outputs/abl/<arm>: D91E and D91B alike
    return roots, gate, ({} if arm == B1_ARM else {"TYPED": f"outputs/abl/{arm}/typed"})


def cols_root(arm):
    """The NEXT columns of the arm's carves: D91E's in the world's own outputs/ltm (zd91m.py ltm cols writes there), D91B's in
    outputs/abl/D91B/ltm (zd91m.py b1 cols)."""
    import ltmcols as LTMC
    return Path(LTMC.OUT) if arm == "D91E" else ROOT / "outputs" / "abl" / B1_ARM / "ltm"


def engine():
    """zswp's read engine with D91's names: its OUT D91's reads folder, the two arms and six datasets added, the columns folder
    per arm. Returns zswp."""
    record_axis()
    import zswp as S
    if getattr(S, "D91R", False):
        return S
    S.OUT, S.STAGE, S.DECLARED = OUT, STAGE, DECLARED
    S.ARMS = tuple(dict.fromkeys(tuple(S.ARMS) + tuple(ARMS)))
    S.DATASETS = tuple(dict.fromkeys(tuple(S.DATASETS) + SIX))
    S.cols_out = cols_root
    S.D91R = True
    return S


# ── fork ─────────────────────────────────────────────────────────────────────


def fork(fit):
    """The fit's folder (the world's outputs/screen/fits) copied without its reads to both arms' fork folders; models.pt
    checked against the fit's D91 record (zd91t: written after a fit that trained)."""
    in_world()
    kind = fit_kind(fit)
    src = SCREEN_FITS / fit
    frec = FIT_RECORDS / f"{fit}.json"
    if not (src / "models.pt").exists():
        raise SystemExit(f"D91R: {rel(src)} holds no models.pt")
    if not frec.exists() or read_json(frec).get("fit") != fit or read_json(frec).get("stage") != STAGE:
        raise SystemExit(f"D91R: {rel(frec)} is not D91's record of {fit} (zd91t.py train writes it)")
    want = sha_file(src / "models.pt")
    if read_json(frec).get("models_sha256") != want:
        raise SystemExit(f"D91R: {rel(src)}'s models.pt is not the one its D91 record names")
    ignore = (lambda d, names: [x for x in names if Path(d) == src and x in ("reads", "read.json")])
    for arm in ARMS:
        dst = fork_dir(arm, fit)
        if (dst / "fork.json").exists():
            if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                raise SystemExit(f"D91R: {rel(dst)} holds other models than {rel(src)}")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp, ignore=ignore)
        if sha_file(tmp / "models.pt") != want:
            raise SystemExit(f"D91R: {rel(tmp)}'s models.pt is not the source's")
        os.replace(tmp, dst)
        write_json(dst / "fork.json", {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "fit": fit, "kind": kind,
                                       "source": rel(src), "models_sha256": want, "d91_fit_record": rel(frec),
                                       "d91_fit_record_sha256": sha_file(frec), "zd91r_sha256": sha_src(__file__), "utc": utc()})
        log(f"fork {fit} -> {rel(dst)}")
    return 0


# ── the reads ────────────────────────────────────────────────────────────────


def read_path(arm, fit, ds):
    return fork_dir(arm, fit) / "reads" / f"{ds}__{check(arm, ds)}.npz"


def marker_path(arm, fit, ds):
    return fork_dir(arm, fit) / f"d91_read_{ds}.json"


def read_zg6r(arm, fit, ds):
    """ML1's read of a zg6r fork on the arm's roots, gate and carve (zabl.read's body: zml1s's code, no stamp)."""
    S = engine()
    import zml1s as ML
    import zp1c3 as Z3
    import zwq as WQ
    carve = check(arm, ds)
    if ML.ALL_FITS.get(fit) is None or ML.ALL_FITS[fit][0] not in ML.BASE_OF:
        raise SystemExit(f"D91R: {fit} is not one of ML1's zg6r fits")
    env = ML.device_env()
    ML.where_ok(env, "read")
    _rec, k = ML.fit_stamp(fit)
    if ML.device_of(env) != k.get("device"):
        raise SystemExit(f"D91R: {fit} was fitted on {k.get('device')}; this read runs on {ML.device_of(env)}")
    roots, gate, typed = roots_gate(ML, arm, ds)
    flags = ["--read", f"{ds}={carve}", "--out-root", rel(OUT / "fits" / arm)]
    with WQ.patch(ML, ROOTS=roots, GATE=gate, **typed), WQ.patch(Z3, eval_carves=lambda: [(ds, carve)]), \
            WQ.patch(Z3.RM, read=S.A.guard_read(Z3.RM.read, flags)):
        rc, extra = ML.run_arm(fit, env, "read")
    return rc, {"device": ML.device_of(env), "gate": gate, "gate_sha256": sha_file(ROOT / gate), "roots": roots, "typed": typed,
                "extra": json.loads(json.dumps(extra, default=str))}


def read_pair(arm, fit, ds):
    """zswp.read (GNX under zgnx.on_gnx, zrckpn under zltm.on_ltm) with the arm's carve, roots, gate and typed links for the
    call: on D91B S.REC is the arm, so no typed links are asked of ABL's folder (passage settings have none)."""
    S = engine()
    carve = check(arm, ds)
    saved = (S.EVAL, S.roots_gate, S.REC)
    roots, gate, _typed = roots_gate(S.M1, arm, ds)
    S.EVAL, S.roots_gate = carve, (lambda _a: (roots, gate))
    if arm == B1_ARM:
        S.REC = arm
    try:
        rc = S.read(arm, fit, ds)
    finally:
        S.EVAL, S.roots_gate, S.REC = saved
    swp = fork_dir(arm, fit) / f"swp_read_{ds}.json"
    return rc, {"gate": gate, "gate_sha256": sha_file(ROOT / gate), "roots": roots,
                "engine_marker": rel(swp), "engine_marker_sha256": sha_file(swp) if swp.exists() else None}


def read(arm, fit, ds):
    """One D91 fit's read of the arm's carve of ds, with the record's model code; D91's marker after."""
    in_world()
    carve = check(arm, ds)
    kind = fit_kind(fit)
    fdir = fork_dir(arm, fit)
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"D91R: {rel(fdir)} is not forked (zd91r.py fork --name {fit})")
    out, marker = read_path(arm, fit, ds), marker_path(arm, fit, ds)
    if marker.exists() and out.exists():
        log(f"read {arm}/{fit}/{ds}: filed")
        return 0
    if kind == "zrckpn" and not _MLP:
        raise SystemExit("D91R: a zrckpn read starts with `zd91r.py read ... --name scr-ltm-zrckpn-*` (zltm's environment at start)")
    t = time.time()
    rc, info = (read_zg6r if kind == "zg6r" else read_pair)(arm, fit, ds)
    ZL = record_axis()
    if rc:
        return rc
    if not out.exists():
        raise SystemExit(f"D91R: the read wrote no {rel(out)}")
    write_json(marker, {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "fit": fit, "kind": kind, "dataset": ds, "carve": carve,
                        "split": fit_split(fit), "read": rel(out), "read_sha256": sha_file(out), "zlink_fams": ZL.FAMS,
                        "fork_sha256": sha_file(fdir / "fork.json"), **info, "seconds": round(time.time() - t, 1),
                        "zd91r_sha256": sha_src(__file__), "utc": utc()})
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
        os.environ.pop("D91WORLD", None)
        for f in (lambda: fork(FITS[0]), lambda: read("D91B", FITS[0], "2wiki"), lambda: read("D91E", FITS[-1], "squad")):
            try:
                f()
                raise AssertionError("must refuse outside the D91 world")
            except SystemExit as e:
                assert "D91 world" in str(e), e
    finally:
        os.environ.clear()
        os.environ.update(keep)
    for a, d in (("D91B", "squad"), ("D91E", "b1v"), ("X", "2wiki"), ("B1V", "2wiki"), ("B1X", "2wiki"), ("D83E", "squad")):
        try:
            check(a, d)
            raise AssertionError(f"{a}/{d} must be refused")
        except SystemExit:
            pass
    # in a fresh process: the engine's names, the roots, gates and typed links per arm, the columns folders, the fits known to
    # ML1, zgnx and zltm, and the record's three families throughout
    code = r"""
import sys, os
sys.path.insert(0, r'%s')
import zd91r as R
S = R.engine()
import zlink as ZL, zml1s as ML, zgnx as GX, zltm as LTM, ltmcols as LTMC
assert ZL.FAMS == 3 and S.OUT == R.OUT and set(R.ARMS) <= set(S.ARMS) and set(R.SIX) <= set(S.DATASETS)
assert 'zd83m' not in sys.modules and not getattr(GX, 'D83_LAB', False) and not getattr(LTM, 'D83_LAB', False)
assert all((d, 's1eval') in LTMC.ALL_CARVES for d in R.SIX)
for f in R.FITS:
    k = R.fit_kind(f)
    assert (f in ML.ALL_FITS and ML.ALL_FITS[f][0] in ML.BASE_OF) if k == 'zg6r' else (f in S.GNX_FITS if k == 'gnx' else f in S.MLP_FITS)
    assert R.fit_split(f) == (ML.ALL_FITS[f][2] if k == 'zg6r' else S.fit_split(f)), f
r, g, t = R.roots_gate(ML, 'D91E', 'metaqa')
assert r['cache'] == 'outputs/abl/D91E/cache' and r['links'] == 'outputs/abl/D91E/links' and g == 'outputs/abl/D91E/gate.json'
assert t == {'TYPED': 'outputs/abl/D91E/typed'}
r, g, t = R.roots_gate(ML, 'D91B', 'musique')
assert r['cache'] == 'outputs/abl/D91B/cache' and r['pools'] == 'outputs/abl/D91B/pools' and r['kt'] == 'outputs/abl/D91B/kt' and g == 'outputs/abl/D91B/gate.json' and t == {}
assert R.cols_root('D91E') == __import__('pathlib').Path(LTMC.OUT) and R.cols_root('D91B') == R.ROOT / 'outputs' / 'abl' / 'D91B' / 'ltm'
assert S.cols_out('D91B') == R.cols_root('D91B') and S.read_path('D91B', R.FITS[0], 'x').parent.parent == R.fork_dir('D91B', R.FITS[0])
R.record_axis()
print('engine OK')
""" % str(HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0 and "engine OK" in r.stdout, (r.stdout[-1500:], r.stderr[-3000:])
    # a zrckpn read's process starts in zltm's environment
    code = (r"import sys; sys.argv = ['zd91r.py', 'read', '--name', 'scr-ltm-zrckpn-s0']; sys.path.insert(0, r'%s'); import os; "
            r"import zd91r; assert os.environ['OMP_NUM_THREADS'] == os.environ['OPENBLAS_NUM_THREADS'] == '6' and "
            r"os.environ['CUDA_VISIBLE_DEVICES'] == '' and zd91r._MLP; print('env OK')" % str(HERE))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0 and "env OK" in r.stdout, r.stderr[-2000:]
    print("zd91r self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zd91r.py")
    ap.add_argument("cmd", choices=("selftest", "fork", "read"))
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--name")
    ap.add_argument("--dataset")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "fork":
        if not a.name:
            raise SystemExit("D91R fork: --name")
        return fork(a.name)
    if not (a.arm and a.name and a.dataset):
        raise SystemExit("D91R read: --arm, --name and --dataset")
    return read(a.arm, a.name, a.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
