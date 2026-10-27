"""PSMK (docs/PSMK_PSM_ON_T4.md): PSM's screen (docs/PSM_POOL_SOFTMAX_INPUT.md) on Kaggle's T4, under WQK's base set
(docs/WQK_WQ_ON_T4.md). zrckpg (zpsm.py, imported unchanged) is trained by zp1c3p's train under zpsm's arm swap at WQK's
names and roots, with WQK's mixed KT input (zwqk.py, imported unchanged), and read on the device it was fitted on. Each fit
is compared with WQK's seed-0 zrckp fit of its split alone, both on one device, and every read is called under WQK's
floors on zrckp by zpsm's own verdict code with PSMK's names.

In the kernel (zpsmk_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zpsmk.py fit --name scr-psmk-zrckpg-hp --rows-cap 65536

On the laptop, after `kaggle kernels output` into a new empty folder:

    python outputs/mp_unified/zpsmk.py take --from DIR   # each fit's records and reads, never its weights; the gate
    python outputs/mp_unified/zpsmk.py compare-all       # outputs/screen/scr-psmk-zrckpg[-hp].json
    python outputs/mp_unified/zpsmk.py verdict           # outputs/psmk/screen-verdict (.json, .md)
    python outputs/mp_unified/zpsmk.py --selftest

This file imports the standard library, zwq and zwqk only; zpsm (torch, zp1c3p and PSM's arm) is imported by the commands
that need it. verdict exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2 after writing an INCOMPLETE record.
"""
import contextlib
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zwqk as K  # noqa: E402  (the standard library and zwq at import)

WQ = K.WQ
DECLARED = "docs/PSMK_PSM_ON_T4.md"
STAGE = "PSMK"
KEY = "psmk"                                  # the stamp's key in screen.json, read.json and PSMK's compares
ARM, BASE = "zrckpg", "zrckp"                 # zpsm's, checked when it is imported
SPLITS = tuple(WQ.SPLITS)
FIT = {sp: f"scr-psmk-{ARM}{WQ.SUF[sp]}" for sp in SPLITS}
SPLIT_OF = {n: sp for sp, n in FIT.items()}
BASE_FIT = {sp: K.fit_name(BASE, 0, sp) for sp in SPLITS}
FLOORS = K.FLOORS[BASE]
VERDICT = "outputs/psmk/screen-verdict"
SESSIONS = "outputs/psmk/sessions"
CODE = K.CODE + ("zpsm", "zpsmk")
RULE = ("ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; "
        "NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMK's, or not "
        "twelve reads (docs/PSMK_PSM_ON_T4.md, section 3)")
MD = (("# PSM's screen verdict:", "# PSMK, PSM's screen on Kaggle's T4:"),
      ("against WQ's card fit of each split, every read called under WQ's floors",
       "against WQK's seed-0 zrckp fit of each split on the same T4, every read called under WQK's floors"))
assert len(set(FIT.values())) == len(FIT) == 2 and all(K.FITS.get(b) == (BASE, 0, sp) for sp, b in BASE_FIT.items())
assert not set(FIT.values()) & set(K.FITS) and FLOORS == "outputs/wqk/floors-zrckp"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpsmk: {msg}", flush=True)


# ── zpsm at PSMK's names ─────────────────────────────────────────────────────


def import_psm():
    """zpsm (torch, zp1c3p and PSM's arm in lean_screen's table), imported outside zwq.on_wq; its arm checked PSMK's."""
    import zpsm as P
    if (P.ARM, P.BASE, tuple(P.SPLITS)) != (ARM, BASE, SPLITS) or set(P.FIT.values()) & set(FIT.values()):
        raise SystemExit(f"zpsmk: zpsm's arm {P.ARM} on {P.SPLITS} is not PSMK's {ARM} on {SPLITS}")
    return P


@contextlib.contextmanager
def on_psmk():
    """zpsm with PSMK's fits, bases, stage, rule, record and declaration, inside zwqk's names (WQK's roots, records, fit
    names and floors); every name restored after, on a failure too."""
    P = import_psm()
    with K.on_wqk(), WQ.patch(P, DECLARED=DECLARED, STAGE=STAGE, FIT=dict(FIT), SPLIT_OF=dict(SPLIT_OF),
                               BASE_FIT=dict(BASE_FIT), VERDICT=VERDICT, RULE=RULE):
        yield P


# ── guards ───────────────────────────────────────────────────────────────────


def split_of(name):
    if name not in SPLIT_OF:
        raise SystemExit(f"zpsmk: {name} is not one of PSMK's fits ({', '.join(FIT.values())})")
    return SPLIT_OF[name]


def where_ok(env, what):
    if not K.ok_where(env):
        raise SystemExit(f"zpsmk: PSMK {what}s in a Kaggle kernel on a {K.GPU} from the image {K.IMAGE_TAG} (section "
                         f"4), not with {K.device_of(env)} ({env.get('where')})")


def wqk_ready(base=None):
    """WQK's verdict record, committed before the stub: BASE_READY under WQK's declaration (section 2)."""
    p = Path(base or WQ.ROOT) / (K.VERDICT + ".json")
    if not p.is_file():
        raise SystemExit(f"zpsmk: {K.VERDICT}.json missing; PSMK trains after WQK's BASE_READY (section 2)")
    rec = WQ.read_json(p)
    if (rec.get("stage"), rec.get("declared_in"), rec.get("verdict")) != (K.STAGE, K.DECLARED, "BASE_READY"):
        raise SystemExit(f"zpsmk: {K.VERDICT}.json is {rec.get('verdict')}, not WQK's BASE_READY")
    return rec


# ── train and read in the kernel (section 4) ─────────────────────────────────


def code_shas():
    return {m: WQ.sha_src(HERE / f"{m}.py") for m in CODE}


def gap_columns(name, P, base=None):
    """zpsm.gap_columns at base: {block: width} of the fit's gap blocks, from its train.json (None before it exists)."""
    tj = WQ.fit_dir(name, Path(base or WQ.ROOT)) / "train.json"
    if not tj.is_file():
        return None
    w = (WQ.read_json(tj).get("variants") or {}).get("p", {}).get("widths") or {}
    return {b: int(v) for b, v in w.items() if b not in P.NO_GAP}


def fit_records(split):
    """The KT and chain records the fit trained on (zwqk.stamp_fit's), at WQK's roots with the mixed KT input."""
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, W.Z2.on_p1c2():
        return {"kt_records": W.Z3.kt_records(W.Z3.train_carves(split)), "rmatch_records": W.Z3.RM.built_records(),
                "zrc_records": W.Z2._BUILT(W.Z2.CH_ZRC), "zlink_records": W.Z3.ZL.built_records()}


def stamp_fit(name, env, P, base=None, records=None):
    """The fit's record after zp1c3p's train: the wq block at WQK's roots and declaration (zwqk.wq_block), PSM's psm block
    naming PSMK, and PSMK's block (the device, WQK's gate, overlay and check, the KT and chain records, the code)."""
    base = Path(base or WQ.ROOT)
    split = split_of(name)
    sj = WQ.fit_dir(name, base) / "screen.json"
    rec = WQ.read_json(sj)
    records = fit_records(split) if records is None else records
    cols = gap_columns(name, P, base)
    ver = base / (K.VERDICT + ".json")
    with WQ.patch(WQ, ROOT=base):
        wq = K.wq_block(ARM, 0, split)
    rec.update({"zwq_sha256": WQ.sha_src(WQ.__file__), "zwqk_sha256": WQ.sha_src(K.__file__),
                "zpsm_sha256": WQ.sha_src(P.__file__), "zpsmk_sha256": WQ.sha_src(__file__), "wq": wq})
    rec["psm"] = {"declared_in": DECLARED, "arm": ARM, "base_arm": BASE, "split": split, "seed": 0,
                  "against": BASE_FIT[split], "input": P.GAP, "gap_columns": cols,
                  "gap_weights": None if cols is None else sum(cols.values()) * int(rec.get("hidden") or 0),
                  "wqk_verdict": WQ.sha_file(ver), "wqk_verdict_utc": WQ.read_json(ver).get("utc")}
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": 0, "split": split, "against": BASE_FIT[split],
                "device": K.device_of(env), "env": env, "gate": WQ.sha_file(base / K.RECORDS["GATE"]),
                "overlay": WQ.sha_file(base / K.RECORDS["OVERLAY"]), "check": WQ.sha_file(base / K.RECORDS["CHECK"]),
                "cache_root": K.ROOTS["cache"], **records, "code": code_shas()}
    WQ.write_json(sj, rec)
    return rec


def train(name, env=None):
    """PSMK's fit (section 4): zp1c3p's train for zrckpg under zpsm's arm swap, at WQK's names and roots with WQK's mixed
    KT input, as zwqk's train runs zrckp; then the fit's stamps."""
    split = split_of(name)
    env = K.device_env() if env is None else env
    where_ok(env, "train")
    wqk_ready()
    P = import_psm()
    with K.on_wqk():
        WQ.gate_passed()
    args = ["train", "--name", name, "--arm", ARM, "--seed", "0", "--device", "cuda", "--cache-root", K.ROOTS["cache"]]
    with K.on_mixed_kt(), on_psmk(), WQ.on_wq() as W, P.on_arm(W):
        with WQ.patch(W.Z3P, FIT={**W.Z3P.FIT, split: name}):
            rc = W.Z3P.train(args, split)
    if rc == 0:
        rec = stamp_fit(name, env, P)
        log(f"train {name}: done, stamped ({sum((rec['psm']['gap_columns'] or {}).values())} gap columns)")
    return rc


def fit_stamp(name, fits_root=None):
    """A PSMK fit's screen.json and its PSMK block, checked against the fit's name."""
    split = split_of(name)
    sj = Path(fits_root or WQ.fit_dir(name).parent) / name / "screen.json"
    rec = WQ.read_json(sj) if sj.is_file() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("split")) != \
            (ARM, DECLARED, name, ARM, split):
        raise SystemExit(f"zpsmk: {name} is not PSMK's {ARM} fit of {split} (no PSMK stamp)")
    return rec, k


def read(name, env=None, rows_cap=K.ROWS_CAP):
    """The fit read by zp1c3p's read for zrckpg with p@swa on the six s1eval carves, --rows-cap 65536, on the device it
    was fitted on (zwqk's read, with zpsm's arm swap). read.json is stamped when the read exits 0."""
    split = split_of(name)
    K.cap_ok(rows_cap)
    env = K.device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if K.device_of(env) != k.get("device"):
        raise SystemExit(f"zpsmk: {name} was fitted on {k.get('device')}; this read runs on {K.device_of(env)}")
    P = import_psm()
    with on_psmk():
        WQ.gate_passed()
        why = P.psm_why(name, WQ.ROOT)
    if why:
        raise SystemExit(f"zpsmk: {why}")
    argv = ["--name", name, "--device", "cuda", "--cache-root", K.ROOTS["cache"], "--rows-cap", str(K.ROWS_CAP)]
    with K.on_mixed_kt(), on_psmk(), WQ.on_wq() as W, P.on_arm(W):
        rc = W.Z3P.read(argv)
        kt = W.Z3.kt_records(W.Z3.eval_carves())
    rj = WQ.fit_dir(name) / "read.json"
    if rc == 0 and rj.is_file():
        rec = WQ.read_json(rj)
        rec["zpsmk_sha256"] = WQ.sha_src(__file__)
        rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": ARM, "split": split, "device": K.device_of(env),
                    "env": env, "gate": WQ.sha_file(WQ.ROOT / K.RECORDS["GATE"]), "kt_records": kt,
                    "rows_cap": int(rows_cap)}
        WQ.write_json(rj, rec)
    return rc


def stamped(p, name):
    try:
        k = WQ.read_json(p).get(KEY) or {}
    except (OSError, ValueError):
        return False
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def fit(name, rows_cap=K.ROWS_CAP):
    """One fit in the kernel (zwqk's fit, for PSMK): a read PSMK stamped is kept; a fit trained and stamped but not read
    is only read; otherwise the fit's files are unlinked and it trains and reads. psmk_timing.json holds the seconds and
    the peak."""
    split_of(name)
    K.cap_ok(rows_cap)
    fd = WQ.fit_dir(name)
    if stamped(fd / "read.json", name):
        log(f"fit {name}: read before, kept")
        return 0
    env = K.device_env()
    where_ok(env, "train")
    wqk_ready()
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").is_file() and (fd / "models.pt").is_file()
    if not trained:
        K.clear(fd, "*")
        K.clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    K.clear(fd / "reads", "*")
    K.clear(fd, "read.json")
    rc = read(name, env, rows_cap)
    WQ.write_json(fd / "psmk_timing.json", {"fit": name, "trained_here": not trained,
                                            "train_seconds": round(t1 - t0, 1),
                                            "read_seconds": round(time.time() - t1, 1), "peak_gb": K.peak_gb(),
                                            "rows_cap": int(rows_cap), "exit": rc, "declared_in": DECLARED})
    return rc


# ── on the laptop: take, compare-all, verdict ────────────────────────────────


def fit_device(name, fits_root=None):
    """A PSMK fit's device: its screen.json's PSMK block, and its read.json's on the same device with --rows-cap 65536."""
    _rec, k = fit_stamp(name, fits_root)
    rj = Path(fits_root or WQ.fit_dir(name).parent) / name / "read.json"
    r = WQ.read_json(rj) if rj.is_file() else {}
    rk = r.get(KEY) or {}
    if (rk.get("declared_in"), rk.get("fit"), rk.get("device"), rk.get("rows_cap")) != \
            (DECLARED, name, k.get("device"), K.ROWS_CAP):
        raise SystemExit(f"zpsmk: {name} has no PSMK read on the device it was fitted on")
    return k["device"]


def pair_device(split, base):
    """The one device PSMK's fit of split and WQK's seed-0 zrckp fit of split were both fitted and read on (section 2)."""
    name, b0 = FIT[split], BASE_FIT[split]
    fr = K.fits_dir(base)
    dn, db = fit_device(name, fr), K.fit_device(b0, fr)
    if dn != db:
        raise SystemExit(f"zpsmk: {name} ran on {dn} and its base {b0} on {db}; a fit and its base are read on one "
                         f"device (section 2)")
    return dn


def take(src, base=None):
    """A kernel's output (src): each PSMK fit copied to outputs/screen/fits/<name>, its records and reads, never its
    weights; the session's gate.json and overlay.json to outputs/psmk/sessions/<the gate's sha256[:12]>. A fit without
    PSMK's stamp and read on one device is left (named); a folder not PSMK's, or a file already there with other bytes,
    stops the take. Returns (copied, left)."""
    base, src = Path(base or WQ.ROOT), Path(src)
    fits = src / "fits"
    g, o = src / "wqk" / "gate.json", src / "wqk" / "overlay.json"
    if not (g.is_file() and o.is_file()):
        raise SystemExit(f"zpsmk take: {src.name} holds no wqk/gate.json and wqk/overlay.json (the session's records)")
    names = sorted(p.name for p in fits.iterdir() if p.is_dir()) if fits.is_dir() else []
    bad = [n for n in names if n not in SPLIT_OF]
    if bad:
        raise SystemExit(f"zpsmk take: {bad} are not PSMK's fits")
    done, left = [], []
    for n in names:
        try:
            fit_device(n, fits)
            done.append(n)
        except SystemExit as e:
            left.append(f"{n}: {e}")
    plan = [(f, K.fits_dir(base) / n / f.relative_to(fits / n)) for n in done
            for f in sorted(p for p in (fits / n).rglob("*") if p.is_file())
            if f.name not in K.NEVER and f.suffix not in (".tmp", ".part")]
    sess = base / SESSIONS / WQ.sha_file(g)[:12]
    plan += [(g, sess / "gate.json"), (o, sess / "overlay.json")]
    for f, t in plan:
        if t.exists() and WQ.sha_file(t) != WQ.sha_file(f):
            raise SystemExit(f"zpsmk take: {WQ.rel(t, base)} exists with other bytes")
    took = []
    for f, t in plan:
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(WQ.rel(t, base))
    log(f"take: {len(done)} fits, {len(took)} files copied; the gate {sess.name}" + (f"; left: {left}" if left else ""))
    return took, left


def compare_all(base=None):
    """zpsm's compare of each PSMK fit with WQK's seed-0 zrckp fit of its split alone, run from the repository's root,
    both on one device; each stamped with PSMK's record and that device. A compare PSMK stamped is kept."""
    base = Path(base or WQ.ROOT)
    P = import_psm()
    done, ran = [], []
    for sp in SPLITS:
        name, b0 = FIT[sp], BASE_FIT[sp]
        js = K.screen_dir(base) / f"{name}.json"
        if js.is_file():
            k = WQ.read_json(js).get(KEY) or {}
            if (k.get("declared_in"), k.get("new"), k.get("bases")) == (DECLARED, name, [b0]):
                done.append(name)
                continue
        dev = pair_device(sp, base)
        with on_psmk():
            rc = P.compare(sp, base)
        if rc or not js.is_file():
            raise SystemExit(f"zpsmk: the compare {name} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": name, "new": name, "bases": [b0], "device": dev,
                    "zpsmk_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(name)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def compare_device(split, base):
    """PSMK's compare of split, checked: stamped PSMK's, of its fit with WQK's seed-0 zrckp fit alone, made on the one
    device both fits ran on. Returns that device."""
    name, b0 = FIT[split], BASE_FIT[split]
    p = K.screen_dir(base) / f"{name}.json"
    if not p.is_file():
        raise SystemExit(f"outputs/screen/{name}.json missing")
    k = WQ.read_json(p).get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, name, name, [b0]):
        raise SystemExit(f"outputs/screen/{name}.json is not PSMK's compare of {name} with {b0}")
    dev = pair_device(split, base)
    if k.get("device") != dev:
        raise SystemExit(f"outputs/screen/{name}.json was made on {k.get('device')}, not on the fits' device {dev}")
    return dev


def verdict(base=None, out=None):
    """PSMK's verdict (section 3): zpsm's verdict code with PSMK's names, each comparison first checked PSMK's and on its
    fits' one device; outputs/psmk/screen-verdict.json and .md."""
    base = Path(base or WQ.ROOT)
    P = import_psm()
    orig_rows, orig_write = P.compare_rows, WQ.write
    devices = {}
    ver = base / (K.VERDICT + ".json")

    def compare_rows(split, base_=base):
        devices[split] = compare_device(split, base_)
        return orig_rows(split, base_)

    def write(path, rec, md):
        rec.update({"devices": dict(devices), "wqk_verdict": WQ.sha_file(ver) if ver.is_file() else None,
                    "zwqk_sha256": WQ.sha_src(K.__file__), "zpsmk_sha256": WQ.sha_src(__file__)})
        for a, b in MD:
            md = [ln.replace(a, b) for ln in md]
        orig_write(path, rec, md)

    with on_psmk(), WQ.patch(P, compare_rows=compare_rows), WQ.patch(WQ, write=write):
        return P.verdict(base, Path(out) if out else base / VERDICT)


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_fit(fits_root, name, dev=None, read_=True, stamp=True, rows_cap=K.ROWS_CAP):
    """A PSMK fit's records (the blocks take, compare and verdict read), its weights and a read."""
    split = SPLIT_OF[name]
    d = Path(fits_root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    k = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": 0, "split": split,
         "device": dev or K.device_of(K.TOY_ENV)}
    s = {"arm": ARM, "seed": 0, "split": split, "rc": 0, "p1c3p": {},
         "wq": {"declared_in": K.DECLARED, "split": split, "seed": 0, "arm": ARM},
         "psm": {"declared_in": DECLARED, "arm": ARM, "split": split, "seed": 0}}
    WQ.write_json(d / "screen.json", {**s, **({KEY: k} if stamp else {})})
    WQ.write_json(d / "train.json", {"seconds": 100.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read_:
        WQ.write_json(d / "read.json", {"seconds": 10.0, "carves": {c: {} for c in K.SIX},
                                        KEY: dict(k, rows_cap=rows_cap)})
    return d


def fake_rows(split, calls, seed0=0.5):
    rows = []
    for ds in WQ.EVAL_ORDER:
        d, c = calls.get((split, ds), (0.001, "WITHIN"))
        rows.append({"dataset": ds, "read": "in-domain" if ds in WQ.SPLITS[split] else "zero-shot", "questions": 100,
                     "base": [seed0, 0.3, 0.2], "new": [round(seed0 + d, 4), 0.3, 0.2], "rrf": [0.4, 0, 0],
                     "delta": [d, 0.002, -0.001], "lo": [d - 0.004, 0, 0], "hi": [d + 0.004, 0, 0], "call": c})
    return rows


def fake_world(base, calls=None, seed0=0.5, dev=None, base_dev=None, over=None):
    """WQK's floors on zrckp, PSMK's two fits and WQK's two seed-0 zrckp fits (base_dev or one device), and PSMK's two
    compares, stamped, in base. calls: {(split, ds): (delta, filed call)}; over: {split: fields replaced in that compare}."""
    calls, over, dev = calls or {}, over or {}, dev or K.device_of(K.TOY_ENV)
    reads = [{"split": sp, "dataset": ds, "read": "x", "seed0": seed0, "d1": 0.0, "d2": 0.0, "spread": 0.0,
              "floor": WQ.FLOOR, "raised": False} for sp in SPLITS for ds in WQ.EVAL_ORDER]
    WQ.write_json(Path(base) / (FLOORS + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "arm": BASE,
                                                    "status": "OK", "reads": reads})
    for sp in SPLITS:
        name, b0 = FIT[sp], BASE_FIT[sp]
        fake_fit(K.fits_dir(base), name, dev)
        K.fake_fit(K.fits_dir(base), b0, "g" * 64, "o" * 64, dev=base_dev or dev)
        b = f"outputs/screen/fits/{b0}"
        rec = {"new": f"outputs/screen/fits/{name}", "bases": [b], "candidate": "p@swa", "carve": "s1eval",
               "trained_on": sorted(WQ.SPLITS[sp]), "by_base": {b: {"rows": fake_rows(sp, calls, seed0)}},
               "stage": STAGE, "declared_in": DECLARED, "decided_against": b0, "split": sp,
               KEY: {"declared_in": DECLARED, "compare": name, "new": name, "bases": [b0], "device": dev}}
        rec.update(over.get(sp, {}))
        WQ.write_json(K.screen_dir(base) / f"{name}.json", rec)


def real_state():
    """What PSMK has under the real folders: its fits, its compares and outputs/psmk."""
    r, p = WQ.ROOT, WQ.ROOT / "outputs" / "psmk"
    return (sorted(x.name for x in K.fits_dir(r).glob("scr-psmk-*")),
            sorted(x.name for x in K.screen_dir(r).glob("scr-psmk-*")),
            sorted(x.relative_to(p).as_posix() for x in p.rglob("*")) if p.is_dir() else [])


def selftest():
    import tempfile
    t0 = time.time()
    before = real_state()
    P = import_psm()
    stops, TOY = K.stops, K.TOY_ENV
    # 1. the names: zpsm's arm on WQ's two splits, distinct from PSM's and WQK's fits; each base WQK's seed-0 zrckp fit of
    #    its split, the floors WQK's; zpsm's and zwq's names PSMK's and WQK's inside on_psmk, restored after
    assert FIT == {"WQ-musique": "scr-psmk-zrckpg", "WQ-hotpotqa": "scr-psmk-zrckpg-hp"}
    assert BASE_FIT == {"WQ-musique": "scr-wqk-zrckp-s0", "WQ-hotpotqa": "scr-wqk-zrckp-s0-hp"}
    assert (P.ARM, P.BASE, tuple(P.SPLITS)) == (ARM, BASE, SPLITS) and P.NEXT["ADOPT_FOR_FULL_RUN"].startswith("zspkg")
    assert not set(FIT.values()) & (set(P.FIT.values()) | set(K.FITS) | set(WQ.FITS))
    assert FLOORS == K.FLOORS[BASE] == "outputs/wqk/floors-zrckp" and VERDICT != P.VERDICT
    assert set(CODE) >= {"zpsm", "zpsmk", "zwqk", "zwq", "zp1c3p", "zp1c3", "zp1c2"}
    assert all((HERE / f"{m}.py").is_file() for m in CODE)
    saved = (P.DECLARED, P.STAGE, P.FIT, P.SPLIT_OF, P.BASE_FIT, P.VERDICT, P.RULE, WQ.DECLARED, WQ.FLOORS, WQ.FITS,
             WQ.ROOTS)
    with on_psmk() as P_:
        assert P_ is P and (P.DECLARED, P.STAGE, P.FIT, P.BASE_FIT, P.VERDICT, P.RULE) == \
            (DECLARED, STAGE, FIT, BASE_FIT, VERDICT, RULE)
        assert (WQ.DECLARED, WQ.FLOORS[BASE], WQ.ROOTS["cache"]) == (K.DECLARED, FLOORS, "outputs/wqk/p1c2/cache")
        assert WQ.FITS is K.FITS and P.SPLIT_OF[FIT["WQ-hotpotqa"]] == "WQ-hotpotqa"
    assert (P.DECLARED, P.STAGE, P.FIT, P.SPLIT_OF, P.BASE_FIT, P.VERDICT, P.RULE, WQ.DECLARED, WQ.FLOORS, WQ.FITS,
            WQ.ROOTS) == saved
    try:
        with on_psmk():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert (P.DECLARED, P.FIT, WQ.DECLARED, WQ.FITS) == (saved[0], saved[2], saved[7], saved[9])
    # 2. the guards: a name not PSMK's, a device that is not a T4 on KG's image, another rows cap; then, on a toy root,
    #    WQK's verdict missing or not BASE_READY, and WQK's gate missing or not PASS
    for bad, has in (({"where": "elsewhere"}, "Kaggle"), ({"gpu": "Tesla P100-PCIE-16GB"}, "T4"),
                     ({"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}, "image")):
        assert has in stops(train, FIT["WQ-musique"], {**TOY, **bad}, has="PSMK trains in a Kaggle kernel")
        stops(read, FIT["WQ-hotpotqa"], {**TOY, **bad}, has="PSMK reads in a Kaggle kernel")
    for n in ("scr-psm-zrckpg", "scr-wqk-zrckp-s0", "scr-psmk-zrckp", "scr-wq-zrckp-s0-hp"):
        stops(train, n, TOY, has="is not one of PSMK's fits")
        stops(read, n, TOY, has="is not one of PSMK's fits")
        stops(fit, n, has="is not one of PSMK's fits")
    for cap in (0, None, 32768):
        stops(fit, FIT["WQ-musique"], rows_cap=cap, has="--rows-cap 65536")
        stops(read, FIT["WQ-musique"], TOY, rows_cap=cap, has="--rows-cap 65536")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        T = td / "world"
        gate0 = WQ.gate_passed
        with WQ.patch(WQ, ROOT=T, fit_dir=lambda name, base=T: Path(base) / "outputs" / "screen" / "fits" / name,
                      gate_passed=lambda base=T: gate0(base)):
            stops(fit, FIT["WQ-musique"], has="PSMK trains in a Kaggle kernel")       # the laptop is not a Kaggle T4
            stops(train, FIT["WQ-musique"], TOY, has="base-verdict.json missing")
            vr = T / (K.VERDICT + ".json")
            WQ.write_json(vr, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "INCOMPLETE"})
            stops(train, FIT["WQ-musique"], TOY, has="is INCOMPLETE, not WQK's BASE_READY")
            WQ.write_json(vr, {"stage": K.STAGE, "declared_in": WQ.DECLARED, "verdict": "BASE_READY"})
            stops(train, FIT["WQ-musique"], TOY, has="not WQK's BASE_READY")         # WQ's declaration, not WQK's
            WQ.write_json(vr, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY", "utc": "t"})
            assert wqk_ready()["verdict"] == "BASE_READY"
            stops(train, FIT["WQ-hotpotqa"], TOY, has="outputs/wqk/gate.json missing")
            WQ.write_json(T / K.RECORDS["GATE"], {"verdict": "FAIL", "declared_in": K.DECLARED})
            stops(train, FIT["WQ-hotpotqa"], TOY, has="is FAIL")
            stops(read, FIT["WQ-hotpotqa"], TOY, has="no PSMK stamp")
        assert WQ.gate_passed is gate0 and WQ.ROOT == ROOT
        # 3. the stamps: the wq block zwqk's at WQK's roots and declaration, PSM's psm block naming PSMK, PSMK's block
        #    with the device, WQK's gate, overlay and check, the KT and chain records and the code; zpsm's own check
        #    (psm_why) passes inside on_psmk only, and a read on another device is refused before anything is read
        WQ.write_json(T / K.RECORDS["GATE"], {"verdict": "PASS", "declared_in": K.DECLARED})
        WQ.write_json(T / K.RECORDS["OVERLAY"], {"status": "OK", "declared_in": K.DECLARED})
        WQ.write_json(T / K.RECORDS["CHECK"], {"status": "REPRODUCED", "declared_in": K.DECLARED})
        recs = {"kt_records": {"webqsp/fit": "k", "2wiki/fit": "u"}, "rmatch_records": {"r": 1}, "zrc_records": {"z": 1},
                "zlink_records": {"l": 1}}
        for sp in SPLITS:
            name = FIT[sp]
            d = T / "outputs" / "screen" / "fits" / name
            WQ.write_json(d / "screen.json", {"arm": ARM, "seed": 0, "split": sp, "rc": 0, "hidden": 16, "p1c3p": {}})
            WQ.write_json(d / "train.json", {"variants": {"p": {"widths": {"rank": 6, "SEMB": 4, "WALK": 3}}}})
            r = stamp_fit(name, TOY, P, T, recs)
            with WQ.patch(WQ, ROOT=T):
                assert r["wq"] == K.wq_block(ARM, 0, sp)
            assert (r["wq"]["declared_in"], r["wq"]["roots"], r["wq"]["trains_on"]) == \
                (K.DECLARED, K.ROOTS, list(WQ.SPLITS[sp]))
            p = r["psm"]
            assert (p["declared_in"], p["arm"], p["split"], p["against"]) == (DECLARED, ARM, sp, BASE_FIT[sp])
            assert p["gap_columns"] == {"rank": 6, "WALK": 3} and p["gap_weights"] == 9 * 16 and p["input"] == P.GAP
            assert p["wqk_verdict"] == WQ.sha_file(T / (K.VERDICT + ".json")) and p["wqk_verdict_utc"] == "t"
            k = r[KEY]
            assert (k["device"], k["env"], k["fit"], k["split"]) == (K.device_of(TOY), TOY, name, sp)
            assert [k[x] for x in ("gate", "overlay", "check")] == [WQ.sha_file(T / K.RECORDS[x]) for x in
                                                                   ("GATE", "OVERLAY", "CHECK")]
            assert all(k[x] == v for x, v in recs.items()) and set(k["code"]) == set(CODE)
            assert k["cache_root"] == K.ROOTS["cache"] and WQ.read_json(d / "screen.json") == r
            assert fit_stamp(name, K.fits_dir(T))[1] == k
            with on_psmk():
                assert P.psm_why(name, T) is None
            assert "is not a PSM fit" in P.psm_why(name, T)                   # outside: PSM's own names
            stops(fit_device, name, K.fits_dir(T), has="has no PSMK read")
            with WQ.patch(WQ, ROOT=T, fit_dir=lambda n, base=T: Path(base) / "outputs" / "screen" / "fits" / n):
                stops(read, name, {**TOY, "cudnn": 1}, has="this read runs on")
        # 4. take: records and reads copied, never models.pt, the gate filed by its sha256; a fit read on another device
        #    and an unstamped fit left; a folder not PSMK's, a file with other bytes, a session without its gate refused
        B, out1 = td / "repo", td / "out1"
        g1, _o1 = K.fake_session(out1, T)
        fake_fit(out1 / "fits", FIT["WQ-musique"])
        fake_fit(out1 / "fits", FIT["WQ-hotpotqa"], read_=False)
        rj = out1 / "fits" / FIT["WQ-hotpotqa"] / "read.json"
        WQ.write_json(rj, {KEY: {"declared_in": DECLARED, "fit": FIT["WQ-hotpotqa"], "rows_cap": K.ROWS_CAP,
                                 "device": {**K.device_of(TOY), "gpu": "Tesla P100"}}})
        took, left = take(out1, B)
        assert len(took) == 4 + 2 and len(left) == 1 and "has no PSMK read on the device" in left[0], (took, left)
        assert not (K.fits_dir(B) / FIT["WQ-musique"] / "models.pt").exists()
        assert (K.fits_dir(B) / FIT["WQ-musique"] / "reads" / "metaqa__s1eval.npz").is_file()
        assert not (K.fits_dir(B) / FIT["WQ-hotpotqa"]).exists()
        assert (B / SESSIONS / g1[:12] / "gate.json").is_file() and take(out1, B) == ([], left)
        fake_fit(out1 / "fits", FIT["WQ-hotpotqa"], stamp=False)
        assert "no PSMK stamp" in take(out1, B)[1][0]
        (K.fits_dir(B) / FIT["WQ-musique"] / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out1, B, has="exists with other bytes")
        shutil.rmtree(K.fits_dir(B) / FIT["WQ-musique"])
        (out1 / "fits" / "scr-wqk-zrckp-s0").mkdir()
        stops(take, out1, B, has="are not PSMK's fits")
        (out1 / "fits" / "scr-wqk-zrckp-s0").rmdir()
        stops(take, td / "out-none", B, has="holds no wqk/gate.json")
        # 5. compare-all (zpsm's compare, lean_screen's faked here) on one device, stamped PSMK's and kept after; refused
        #    when a fit and its base ran on two devices
        C = td / "c"
        fake_world(C)
        for sp in SPLITS:
            (K.screen_dir(C) / f"{FIT[sp]}.json").unlink()

        def fake_run(new, base_fit, out):
            n, b0 = Path(new).name, Path(base_fit).name
            sp = SPLIT_OF[n]
            bp = f"outputs/screen/fits/{b0}"
            WQ.write_json(C / (out + ".json"), {"new": new, "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
                                                "trained_on": sorted(WQ.SPLITS[sp]),
                                                "by_base": {bp: {"rows": fake_rows(sp, {})}}})
            return 0

        with WQ.patch(WQ, run_compare=fake_run):
            assert compare_all(C) == 0
            got = {sp: WQ.read_json(K.screen_dir(C) / f"{FIT[sp]}.json") for sp in SPLITS}
            assert compare_all(C) == 0 and got == {sp: WQ.read_json(K.screen_dir(C) / f"{FIT[sp]}.json") for sp in SPLITS}
        for sp, r in got.items():
            assert (r["stage"], r["declared_in"], r["decided_against"], r["split"]) == (STAGE, DECLARED, BASE_FIT[sp], sp)
            assert r[KEY] == {"declared_in": DECLARED, "compare": FIT[sp], "new": FIT[sp], "bases": [BASE_FIT[sp]],
                              "device": K.device_of(TOY), "zpsmk_sha256": WQ.sha_src(__file__)}
        v = verdict(C, C / "v")
        assert v["verdict"] == "NOT_ADOPTED" and "zrckph" in v["next"] and len(v["reads"]) == 12, v["problems"]
        D = td / "d"
        fake_world(D, base_dev={**K.device_of(TOY), "gpu": "Tesla P100"})
        (K.screen_dir(D) / f"{FIT['WQ-musique']}.json").unlink()
        with WQ.patch(WQ, run_compare=fake_run):
            stops(compare_all, D, has="a fit and its base are read on one device")
        # 6. the verdict, zpsm's code with PSMK's names: a GAIN and no LOSS adopts; a LOSS does not; INCOMPLETE for a
        #    missing compare, floors not WQK's, a base R@5 off the floors' seed-0 value, eleven reads, a compare not
        #    PSMK's, made on another device, or of fits on two devices; zpsm's names restored after
        gain = {("WQ-musique", "webqsp"): (0.02, "GAIN")}
        loss = {**gain, ("WQ-hotpotqa", "metaqa"): (-0.03, "LOSS")}
        five = [r for r in fake_rows("WQ-hotpotqa", {}) if r["dataset"] != "squad"]
        b_hp = f"outputs/screen/fits/{BASE_FIT['WQ-hotpotqa']}"
        cases = ((gain, {}, None, "ADOPT_FOR_FULL_RUN", "zspkg"), (loss, {}, None, "NOT_ADOPTED", "zrckpu"),
                 (gain, {"WQ-hotpotqa": {"by_base": {b_hp: {"rows": five}}}}, None, "INCOMPLETE", "not the six"),
                 (gain, {"WQ-hotpotqa": {KEY: {}}}, None, "INCOMPLETE", "is not PSMK's compare"),
                 (gain, {"WQ-musique": {"stage": "PSM"}}, None, "INCOMPLETE", "is not PSM's comparison"),
                 (gain, {"WQ-musique": {KEY: {"declared_in": DECLARED, "compare": FIT["WQ-musique"],
                                              "new": FIT["WQ-musique"], "bases": [BASE_FIT["WQ-musique"]],
                                              "device": {"gpu": "Tesla P100"}}}}, None, "INCOMPLETE",
                  "not on the fits' device"),
                 (gain, {}, {**K.device_of(TOY), "gpu": "Tesla P100"}, "INCOMPLETE", "read on one device"))
        for i, (calls, over, bdev, want, has) in enumerate(cases):
            W_ = td / f"v{i}"
            fake_world(W_, calls, over=over, base_dev=bdev)
            v = verdict(W_, W_ / "verdict")
            assert v["verdict"] == want, (i, v["verdict"], v["problems"])
            assert has in (v["next"] if want != "INCOMPLETE" else " ".join(v["problems"])), (i, has, v["next"],
                                                                                              v["problems"])
            md = (W_ / "verdict.md").read_text(encoding="utf-8")
            assert md.startswith("# PSMK, PSM's screen on Kaggle's T4:") and v["stage"] == STAGE
            assert (v["floors"], v["against"], v["fits"]) == (FLOORS + ".json", BASE_FIT, FIT)
            if want != "INCOMPLETE":
                assert "under WQK's floors" in md and "WQ's" not in md and len(v["reads"]) == 12
                assert v["devices"] == {sp: K.device_of(TOY) for sp in SPLITS} and v["rule"] == RULE
        W_ = td / "v7"
        fake_world(W_, gain)
        WQ.write_json(W_ / (FLOORS + ".json"), {**WQ.read_json(W_ / (FLOORS + ".json")), "declared_in": WQ.DECLARED})
        v = verdict(W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("not WQ's OK floors" in p for p in v["problems"]), v["problems"]
        W_ = td / "v8"
        fake_world(W_, gain)
        fl = WQ.read_json(W_ / (FLOORS + ".json"))
        fl["reads"][3]["seed0"] = 0.4999
        WQ.write_json(W_ / (FLOORS + ".json"), fl)
        v = verdict(W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("seed-0 R@5" in p for p in v["problems"]), v["problems"]
        (K.screen_dir(W_) / f"{FIT['WQ-musique']}.json").unlink()
        v = verdict(W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("missing" in p for p in v["problems"]), v["problems"]
        assert (P.DECLARED, P.FIT, P.compare_rows.__name__, WQ.write.__name__, WQ.DECLARED) == \
            (saved[0], saved[2], "compare_rows", "write", saved[7])
        assert P.compare_rows.__module__ == "zpsm" and WQ.write.__module__ == "zwq"
    # 7. nothing written under PSMK's real folders
    assert real_state() == before, (real_state(), before)
    log(f"selftest: the names (zpsm's arm on WQ's splits, WQK's seed-0 zrckp bases and floors, restored after); the "
        f"guards (name, device, rows cap, WQK's verdict and gate); the stamps (zwqk's wq block, the psm block naming "
        f"PSMK, PSMK's block); take; compare-all on one device; the verdict's ten cases ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog="zpsmk.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=K.ROWS_CAP)
    ap.add_argument("--from", dest="src")
    a = ap.parse_args(argv)
    src = Path(a.src).resolve() if a.src else None
    os.chdir(WQ.ROOT)
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    if a.cmd == "take":
        if src is None:
            ap.error("take takes --from")
        take(src)
        return 0
    if a.cmd == "compare-all":
        return compare_all()
    return 0 if verdict()["verdict"] != "INCOMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
