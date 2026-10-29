"""R2WS (docs/R2WS_R2_SEEDS_UNDER_WQ.md): R2's bridge head on zrckp (zrckpb, R2GW's beside arm, unchanged) fitted again under
WQ on Kaggle's T4 at WQK's null seeds 1 and 2, on both splits, and compared with WQK's zrckp fit of the same seed, split and
device. R2GW read zrckpb at seed 0 (3 GAIN, 0 LOSS against WQK's seed-0 zrckp) as a beside base before this file was
written; those reads decide nothing here and seed 0 is not in the rule. zrckpb is trained by zp1c3p's train under an arm
swap with zr2br's EM term (zr2gw.py's, imported unchanged), at WQK's names and roots with WQK's mixed KT input (zwqk.py,
imported unchanged), as WQK trains zrckp, with --seed 1 or 2 by the fit's name; read on the device it was fitted on.
Each seed's re-call against zrckp of that seed under WQK's zrckp seed null (its floors, the seed-0 R@5 of each read
replaced by the same-seed zrckp read) is called by lean_screen's compare and screen_recall as R2GW's was, and the rule
joins the two seeds.

In the kernel (zr2ws_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zr2ws.py fit --name scr-r2ws-zrckpb-s2-hp --rows-cap 65536

On the laptop, after `kaggle kernels output` of the session into a new folder:

    python outputs/mp_unified/zr2ws.py take --from DIR   # each fit's records and reads, never its weights
    python outputs/mp_unified/zr2ws.py compare-all       # outputs/screen/scr-r2ws-zrckpb-s<k>-vs-zrckp-s<k>[-hp].json
    python outputs/mp_unified/zr2ws.py verdict           # outputs/r2ws/screen-verdict (.json, .md)
    python outputs/mp_unified/zr2ws.py --selftest

This file imports the standard library, zwqk (with it zwq) and zr2gw only; zr2br and torch are imported by the commands
that need them, outside zwq.on_wq. verdict exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2 after writing an INCOMPLETE
record.
"""
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zr2gw as GW  # noqa: E402  (the standard library, zwq and zwqk at import)

K, WQ = GW.K, GW.WQ
DECLARED = "docs/R2WS_R2_SEEDS_UNDER_WQ.md"
STAGE = "R2WS"
KEY = "r2ws"                                  # the stamp's key in screen.json, read.json and R2WS's compares
ARM, BASE = GW.BESIDE_ARM, GW.BASE            # R2GW's zrckpb, on WQK's zrckp
SPLITS = tuple(GW.SPLITS)
SEEDS = tuple(WQ.NULL_SEEDS)                  # WQK's null seeds: the zrckp fits at seeds 1 and 2 that made the floors
READ_CAP = GW.READ_CAP                        # WQK's rows a read batch (whole questions, so the scores do not change)
NB = GW.NB
R2_N, R2_CONST, R2_PARAMS, R2_DECLARED = GW.R2_N, GW.R2_CONST, GW.R2_PARAMS, GW.R2_DECLARED
FIT = {(sd, sp): f"scr-r2ws-{ARM}-s{sd}{WQ.SUF[sp]}" for sd in SEEDS for sp in SPLITS}
FITS = {n: k for k, n in FIT.items()}         # name -> (seed, split)
BASE_FIT = {(sd, sp): K.fit_name(BASE, sd, sp) for sd in SEEDS for sp in SPLITS}      # WQK's zrckp of the same seed: the base
SEED0_FIT = dict(GW.BASE_FIT)                 # WQK's seed-0 zrckp per split: the floors' own seed 0 (named, not compared)
BESIDE_FIT = {sp: GW.FIT[(ARM, sp)] for sp in SPLITS}      # R2GW's seed-0 zrckpb (read before this file; beside only)
ZSPK_CMP = dict(GW.ZSPK_CMP)
CMP = {(sd, sp): f"scr-r2ws-{ARM}-s{sd}-vs-zrckp-s{sd}{WQ.SUF[sp]}" for sd in SEEDS for sp in SPLITS}
COMPARES = {CMP[k]: (FIT[k], [BASE_FIT[k]]) for k in FIT}
NEW = "outputs/r2ws/"
SESSIONS = NEW + "sessions"
VERDICT = NEW + "screen-verdict"
CALLS = {f"zrckpb-s{sd}": sd for sd in SEEDS}                      # call -> the seed it re-calls
PAIR = {c: f"{NEW}r2ws-{c}-pair" for c in CALLS}
RECALL = {c: f"{p}-recall" for c, p in PAIR.items()}
NULL_KIND = "WQK's zrckp seed null (its floors), each read's seed-0 R@5 replaced by the same-seed zrckp read"
FLOORS = K.FLOORS[BASE]
TIMING_FILE = "r2ws_timing.json"
CODE = tuple(dict.fromkeys(GW.CODE + ("zr2ws",)))
WHAT = "zrckp + R2's bridge head"
NAMING = GW.NAMING_R2T + "; no claim sets zrckpb against the GNN as the MLP, and none rests on the head until that ruling"
SEED0_BESIDE = GW.RECALL[ARM] + ".json"       # R2GW's filed re-call of its seed-0 zrckpb (read as a file, beside only)
RULE = ("ADOPT_FOR_FULL_RUN when, over the seeds fitted (1 and 2), no read of zrckpb's re-call against WQK's zrckp of the "
        "same seed under WQK's zrckp seed null is a LOSS at any seed and at least one read is a GAIN at every seed (the same "
        "read); NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not BASE_READY, a fit or compare is missing, "
        "not R2WS's or of another seed or split than its name, R2WS's fits and the WQK zrckp fits they are compared with "
        "are not on one device, a fit or read names a gate that is not a filed PASS gate of an R2WS session, a fit's R2 "
        "constants or EM term are not as declared or its head names a gate, a compare's zrckp column is not WQK's read of "
        "that seed, WQK's floors are not their compares', or a re-call does not hold the twelve reads "
        "(docs/R2WS_R2_SEEDS_UNDER_WQ.md section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zrckpb fitted again on PKO's pools (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, outputs/pko), then its "
                              "full run under WQ beside zrckp's, each its own declared file; the user's naming ruling is "
                              "still pending, so no claim set rests on the head until they rule whether reading the five "
                              "bridges' vectors keeps R2 inside the MLP track (R2, section 2)",
        "NOT_ADOPTED": "the seed-0 GAINs are called noise (they did not reproduce under this rule); zrckpb is not taken "
                       "further and the R2 line goes on with R2e (R2EW, docs/R2EW_R2E_UNDER_WQ.md) only, whose GNN half "
                       "stays its own later file",
        "INCOMPLETE": "what is missing is supplied (WQK's base set, a fit, a compare, a session's gate or a re-call); "
                      "nothing else changes"}
assert len(FIT) == len(FITS) == 4 and not set(FITS) & (set(K.FITS) | set(GW.FITS))
assert SEEDS == (1, 2) and all(K.FITS.get(b) == (BASE, sd, sp) for (sd, sp), b in BASE_FIT.items())
assert all(b in K.COMPARES and K.COMPARES[b] == (b, [SEED0_FIT[sp]]) for (_sd, sp), b in BASE_FIT.items())
assert all(c not in GW.COMPARES and c not in K.COMPARES for c in COMPARES) and FLOORS == GW.FLOORS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zr2ws: {msg}", flush=True)


# ── guards ───────────────────────────────────────────────────────────────────


def arm_seed_split(name):
    if name not in FITS:
        raise SystemExit(f"zr2ws: {name} is not one of R2WS's fits ({', '.join(FIT.values())})")
    sd, sp = FITS[name]
    return ARM, sd, sp


def where_ok(env, what):
    if not K.ok_where(env):
        raise SystemExit(f"zr2ws: R2WS {what}s in a Kaggle kernel on a {K.GPU} from the image {K.IMAGE_TAG} (section "
                         f"3), not with {K.device_of(env)} ({env.get('where')})")


def cap_ok(rows_cap):
    if rows_cap is None or int(rows_cap) != READ_CAP:
        raise SystemExit(f"zr2ws: R2WS reads with --rows-cap {READ_CAP} (WQK's, section 3), not {rows_cap}")


def wqk_ready(base=None):
    """WQK's verdict record, committed before the stub: BASE_READY under WQK's declaration (section 3)."""
    p = Path(base or WQ.ROOT) / (K.VERDICT + ".json")
    if not p.is_file():
        raise SystemExit(f"zr2ws: {K.VERDICT}.json missing; R2WS trains after WQK's BASE_READY (section 3)")
    rec = WQ.read_json(p)
    if (rec.get("stage"), rec.get("declared_in"), rec.get("verdict")) != (K.STAGE, K.DECLARED, "BASE_READY"):
        raise SystemExit(f"zr2ws: {K.VERDICT}.json is {rec.get('verdict')}, not WQK's BASE_READY")
    return rec


def term_ok(summ, name):
    """A fit whose training had no batch with R2's term is not R2WS's (section 3): it stops unstamped."""
    if not summ.get("batches_with_term"):
        raise SystemExit(f"zr2ws: no training batch of {name} carried R2's term ({summ.get('batches')} batches); unstamped")


# ── train and read in the kernel (section 3) ─────────────────────────────────


def code_shas():
    return {m: WQ.sha_src(HERE / f"{m}.py") for m in CODE}


def train_args(name):
    """zp1c3p's train arguments for a fit: the arm, the seed of the fit's name, WQK's cache root."""
    arm, sd, _sp = arm_seed_split(name)
    return ["train", "--name", name, "--arm", arm, "--seed", str(sd), "--device", "cuda", "--cache-root", K.ROOTS["cache"]]


def stamp_fit(name, env, R2, summ, base=None, records=None):
    """The fit's record after zp1c3p's train: the wq block at WQK's roots and declaration (zwqk.wq_block, with the fit's
    seed), and R2WS's block (the device, WQK's gate, overlay, check and verdict, the KT and chain records, R2's constants,
    parameters and EM term, the code)."""
    base = Path(base or WQ.ROOT)
    arm, seed, split = arm_seed_split(name)
    sj = WQ.fit_dir(name, base) / "screen.json"
    rec = WQ.read_json(sj)
    records = GW.fit_records(split) if records is None else records
    with WQ.patch(WQ, ROOT=base):
        wq = K.wq_block(arm, seed, split)
    rec.update({"zwq_sha256": WQ.sha_src(WQ.__file__), "zwqk_sha256": WQ.sha_src(K.__file__),
                "zr2br_sha256": WQ.sha_src(HERE / "zr2br.py"), "zr2gw_sha256": WQ.sha_src(HERE / "zr2gw.py"),
                "zr2ws_sha256": WQ.sha_src(__file__), "wq": wq})
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "what": WHAT, "seed": seed, "split": split,
                "against": BASE_FIT[(seed, split)], "seed0_base": SEED0_FIT[split], "beside": BESIDE_FIT[split],
                "device": K.device_of(env), "env": env, "gate": WQ.sha_file(base / K.RECORDS["GATE"]),
                "overlay": WQ.sha_file(base / K.RECORDS["OVERLAY"]), "check": WQ.sha_file(base / K.RECORDS["CHECK"]),
                "wqk_verdict": WQ.sha_file(base / (K.VERDICT + ".json")), "cache_root": K.ROOTS["cache"], **records,
                "r2": {"declared_in": R2.DECLARED, "constants": R2.constants(), "params": list(R2.R2_PARAMS),
                       "settings": R2.SETTINGS, "term": summ},
                "head": GW.head_of(arm), "code": code_shas()}
    WQ.write_json(sj, rec)
    return rec


def train(name, env=None):
    """R2WS's fit (section 3): zp1c3p's train for zrckpb under the arm swap with R2's EM term (LAM 1), at WQK's names and
    roots with WQK's mixed KT input and the fit's --seed, as zwqk's train runs zrckp; then the fit's stamps."""
    arm, seed, split = arm_seed_split(name)
    env = K.device_env() if env is None else env
    where_ok(env, "train")
    wqk_ready()
    R2 = GW.import_r2()["R2"]
    with K.on_wqk():
        WQ.gate_passed()
    args = train_args(name)
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, GW.on_arm(W, arm):
        GW.same_modules(R2, W)
        with WQ.patch(W.Z3P, FIT={**W.Z3P.FIT, split: name}), R2.r2_em():
            rc = W.Z3P.train(args, split)
    summ = R2.summary()
    if rc == 0:
        term_ok(summ, name)
        stamp_fit(name, env, R2, summ)
        ep = summ.get("epochs") or []
        log(f"train {name}: done, seed {seed}, stamped (R2's term on {summ['questions_with_term']} of {summ['questions']} "
            f"questions" + (f"; L_br {ep[0]['br_nll_mean']:.2f} -> {ep[-1]['br_nll_mean']:.2f})" if ep else ")"))
    return rc


def fit_stamp(name, fits_root=None):
    """An R2WS fit's screen.json and its R2WS block, checked against the fit's name, seed and split."""
    arm, seed, split = arm_seed_split(name)
    sj = Path(fits_root or WQ.fit_dir(name).parent) / name / "screen.json"
    rec = WQ.read_json(sj) if sj.is_file() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), rec.get("seed"), k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("seed"), k.get("split")) != \
            (arm, seed, DECLARED, name, arm, seed, split):
        raise SystemExit(f"zr2ws: {name} is not R2WS's {arm} fit of {split} at seed {seed} (no R2WS stamp)")
    return rec, k


def stamp_read(name, env, kt, rows_cap, base=None):
    """read.json's R2WS block: the device, the gate and the KT records."""
    base = Path(base or WQ.ROOT)
    arm, seed, split = arm_seed_split(name)
    rj = WQ.fit_dir(name, base) / "read.json"
    rec = WQ.read_json(rj)
    rec["zr2ws_sha256"] = WQ.sha_src(__file__)
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": seed, "split": split,
                "device": K.device_of(env), "env": env, "gate": WQ.sha_file(base / K.RECORDS["GATE"]),
                "kt_records": kt, "rows_cap": int(rows_cap)}
    WQ.write_json(rj, rec)
    return rec


def read(name, env=None, rows_cap=READ_CAP):
    """The fit read by zp1c3p's read for its arm with p@swa on the six s1eval carves, --rows-cap 65536, on the device it was
    fitted on (zwqk's read, under the arm swap). read.json is stamped when the read exits 0."""
    arm, _seed, _split = arm_seed_split(name)
    cap_ok(rows_cap)
    env = K.device_env() if env is None else env
    where_ok(env, "read")
    rec, k = fit_stamp(name)
    if K.device_of(env) != k.get("device"):
        raise SystemExit(f"zr2ws: {name} was fitted on {k.get('device')}; this read runs on {K.device_of(env)}")
    got = (k.get("r2") or {}).get("constants")
    if got != R2_CONST:
        raise SystemExit(f"zr2ws: {name} was fitted with R2 constants {got}, not R2's declared {R2_CONST}")
    R2 = GW.import_r2()["R2"]
    with K.on_wqk():
        WQ.cache_root_ok(K.ROOTS["cache"])
        WQ.gate_passed()
    argv = ["--name", name, "--device", "cuda", "--cache-root", K.ROOTS["cache"], "--rows-cap", str(READ_CAP)]
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, GW.on_arm(W, arm):
        GW.same_modules(R2, W)
        rc = W.Z3P.read(argv)
        kt = W.Z3.kt_records(W.Z3.eval_carves())
    rj = WQ.fit_dir(name) / "read.json"
    if rc == 0 and rj.is_file():
        stamp_read(name, env, kt, rows_cap)
    return rc


def stamped(p, name):
    try:
        k = WQ.read_json(p).get(KEY) or {}
    except (OSError, ValueError):
        return False
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def fit(name, rows_cap=READ_CAP):
    """One fit in the kernel (zwqk's fit, for R2WS): a read R2WS stamped is kept; a fit trained and stamped but not read is
    only read; otherwise the fit's files are unlinked and it trains and reads. r2ws_timing.json holds the seconds and the peak."""
    arm_seed_split(name)
    cap_ok(rows_cap)
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
    WQ.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(time.time() - t1, 1), "peak_gb": K.peak_gb(),
                                     "rows_cap": int(rows_cap), "exit": rc, "declared_in": DECLARED})
    return rc


# ── on the laptop, after the sessions: take, compare-all, verdict (sections 3 and 4) ──


def fit_device(name, fits_root=None):
    """An R2WS fit's device: its screen.json's R2WS block, and its read.json's on the same device with --rows-cap 65536."""
    _rec, k = fit_stamp(name, fits_root)
    rj = Path(fits_root or WQ.fit_dir(name).parent) / name / "read.json"
    r = WQ.read_json(rj) if rj.is_file() else {}
    rk = r.get(KEY) or {}
    if (rk.get("declared_in"), rk.get("fit"), rk.get("seed"), rk.get("device"), rk.get("rows_cap")) != \
            (DECLARED, name, k.get("seed"), k.get("device"), READ_CAP):
        raise SystemExit(f"zr2ws: {name} has no R2WS read on the device it was fitted on")
    return k["device"]


def any_device(name, fits_root):
    """A fit's device by whose it is: R2WS's through fit_device, WQK's through zwqk's."""
    return fit_device(name, fits_root) if name in FITS else K.fit_device(name, fits_root)


def pair_device(names, base):
    """The one device the fits of names were fitted and read on."""
    fr = K.fits_dir(base)
    devs = {n: any_device(n, fr) for n in names}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zr2ws: {names[0]} and its base ran on different devices or images: {devs}")
    return devs[names[0]]


def seed_split_device(sd, sp, base):
    """The one device R2WS's fit of seed sd on sp and WQK's zrckp fit of the same seed and split ran on."""
    return pair_device([FIT[(sd, sp)], BASE_FIT[(sd, sp)]], base)


def take(src, base=None):
    """A kernel's output (src): each R2WS fit copied to outputs/screen/fits/<name>, its records and reads, never its
    weights; the session's gate.json and overlay.json to outputs/r2ws/sessions/<the gate's sha256[:12]>. A fit without
    R2WS's stamp and read on one device is left (named); a folder not R2WS's, or a file already there with other bytes,
    stops the take. Returns (copied, left)."""
    base, src = Path(base or WQ.ROOT), Path(src)
    fits = src / "fits"
    g, o = src / "wqk" / "gate.json", src / "wqk" / "overlay.json"
    if not (g.is_file() and o.is_file()):
        raise SystemExit(f"zr2ws take: {src.name} holds no wqk/gate.json and wqk/overlay.json (the session's records)")
    names = sorted(p.name for p in fits.iterdir() if p.is_dir()) if fits.is_dir() else []
    bad = [n for n in names if n not in FITS]
    if bad:
        raise SystemExit(f"zr2ws take: {bad} are not R2WS's fits")
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
            raise SystemExit(f"zr2ws take: {WQ.rel(t, base)} exists with other bytes")
    took = []
    for f, t in plan:
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(WQ.rel(t, base))
    log(f"take: {len(done)} fits, {len(took)} files copied; the gate {sess.name}" + (f"; left: {left}" if left else ""))
    return took, left


def compare_all(base=None):
    """lean_screen's compare of each R2WS fit with WQK's zrckp fit of the same seed and split, run from the repository's
    root by zwq's run_compare, both on one device; each stamped with R2WS's record and that device. A compare R2WS
    stamped from the same fits is kept."""
    base = Path(base or WQ.ROOT)
    GW.import_r2()
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        js = K.screen_dir(base) / f"{cname}.json"
        if js.is_file():
            k = WQ.read_json(js).get(KEY) or {}
            if (k.get("declared_in"), k.get("new"), k.get("bases")) == (DECLARED, new, list(bases)):
                done.append(cname)
                continue
        dev = pair_device([new] + list(bases), base)
        with K.on_wqk():
            for f in bases:
                why = WQ.wq_why(f, base)
                if why:
                    raise SystemExit(f"zr2ws: {why}")
            rc = WQ.run_compare(f"outputs/screen/fits/{new}", ",".join(f"outputs/screen/fits/{b}" for b in bases),
                                f"outputs/screen/{cname}")
        if rc or not js.is_file():
            raise SystemExit(f"zr2ws: the compare {cname} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                    "seed": FITS[new][0], "zr2ws_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """An R2WS compare's rows {dataset: row} (its one base: WQK's zrckp of the same seed), after its checks; SystemExit
    naming what is wrong."""
    p = Path(js)
    if not p.is_file():
        raise SystemExit(f"outputs/screen/{cname}.json missing")
    rec = WQ.read_json(p)
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases"), k.get("seed")) != \
            (DECLARED, cname, new, list(bases), FITS[new][0]):
        raise SystemExit(f"outputs/screen/{cname}.json is not R2WS's compare of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"outputs/screen/{cname}.json was made on {k.get('device')}, not on the fits' device {dev}")
    got = [WQ.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if WQ.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"outputs/screen/{cname}.json compares {rec.get('new')} with {rec.get('bases')}")
    split = FITS[new][1]
    if sorted(rec.get("trained_on") or []) != sorted(WQ.SPLITS[split]):
        raise SystemExit(f"outputs/screen/{cname}.json trained on {rec.get('trained_on')}, not {split}'s")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"outputs/screen/{cname}.json reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on "
                         f"s1eval")
    rows = {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
    if sorted(rows) != sorted(WQ.EVAL_ORDER):
        raise SystemExit(f"outputs/screen/{cname}.json reads {sorted(rows)} against {bases[0]}, not the six datasets")
    return rec, rows


def filed_gates(base):
    """zwqk.filed_gates on R2WS's sessions: ({gate sha256: its record}, problems), each a WQK PASS gate on the reproduced
    check and its session's overlay."""
    with WQ.patch(K, SESSIONS=SESSIONS):
        return K.filed_gates(base)


def fit_problems(base, gates):
    """Why an R2WS fit is not section 3's: its train exit, its seed in the record, WQK's wq block and the read, its six
    reads, the gates its train, wq block and read name, its overlay, R2's constants and EM term, and that its head names no
    gate (zrckpb is not gated)."""
    out = []
    fr = K.fits_dir(base)
    for (sd, _sp), n in FIT.items():
        sj, rj = fr / n / "screen.json", fr / n / "read.json"
        if not (sj.is_file() and rj.is_file()):
            continue                                  # fit_device names it
        s, r = WQ.read_json(sj), WQ.read_json(rj)
        k, rk = s.get(KEY) or {}, r.get(KEY) or {}
        if s.get("rc") != 0:
            out.append(f"{n}: its train did not exit 0")
        if (s.get("seed"), (s.get("wq") or {}).get("seed"), k.get("seed"), rk.get("seed")) != (sd, sd, sd, sd):
            out.append(f"{n}: its record names seeds {(s.get('seed'), (s.get('wq') or {}).get('seed'), k.get('seed'), rk.get('seed'))}, not {sd}")
        if set(r.get("carves") or {}) != set(K.SIX):
            out.append(f"{n}: not read on the six s1eval carves")
        for what, g in (("train", k.get("gate")), ("wq block", (s.get("wq") or {}).get("gate")), ("read", rk.get("gate"))):
            if g not in gates:
                out.append(f"{n}: its {what} names the gate {str(g)[:12]}, not a filed R2WS PASS gate")
        if k.get("gate") in gates and (gates[k["gate"]].get("records") or {}).get("overlay") != k.get("overlay"):
            out.append(f"{n}: its overlay is not its gate's")
        r2 = k.get("r2") or {}
        term = r2.get("term") or {}
        if (r2.get("declared_in"), r2.get("constants"), r2.get("params"), term.get("lam")) != \
                (R2_DECLARED, R2_CONST, list(R2_PARAMS), 1.0) or not term.get("questions_with_term"):
            out.append(f"{n}: R2's constants or EM term are not as declared")
        if (k.get("head") or {}).get("r2g") is not None:
            out.append(f"{n}: zrckpb is gated (its head names a gate)")
    return out


def column_problems(base, dev):
    """Why a compare's zrckp column is not WQK's read of that seed: its base R@5 against the new column of WQK's own compare
    `scr-wqk-zrckp-s<k>[-hp]`, on the twelve reads. Unreadable compares are named by the verdict's own checks."""
    out = []
    scr = K.screen_dir(base)
    for (sd, sp), cn in CMP.items():
        b = BASE_FIT[(sd, sp)]
        try:
            rows = cmp_rows(scr / f"{cn}.json", cn, dev)[1]
            wqk = K.cmp_rows(scr / f"{b}.json", b, dev)[1]
        except SystemExit:
            continue
        bad = [ds for ds in WQ.EVAL_ORDER if rows[ds]["base"][0] != wqk[ds]["new"][0]]
        if bad:
            out.append(f"{cn}: its zrckp column is not WQK's read of {b} on {', '.join(bad)}")
    return out


def restamp(pair, rcl, c):
    """R2WS's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    sd = CALLS[c]
    for p in (pair, rcl):
        js = Path(str(p) + ".json")
        rec = WQ.read_json(js)
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "seed": sd, "against": f"{BASE} (seed {sd})",
                    "null_kind": NULL_KIND, "deciding": True, "zr2ws_sha256": WQ.sha_src(__file__)})
        WQ.write_json(js, rec)
        md = Path(str(p) + ".md")
        if md.is_file():
            md.write_text(f"R2WS ({DECLARED}): {ARM} at seed {sd} against {BASE} at seed {sd} under WQ on the T4, with "
                          f"the rule joining the seeds.\n\n" + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(base, dev, c):
    """screen_pair over the seed's two compares (their one base: WQK's zrckp of that seed), then screen_recall under WQK's
    zrckp null: its floors, each read's seed-0 R@5 replaced by the same-seed zrckp read of the compare (the check that it
    is WQK's read is the verdict's column_problems)."""
    import screen_pair as SP
    import screen_recall as SR
    base = Path(base)
    scr = K.screen_dir(base)
    sd = CALLS[c]

    def load(fn):
        cn = next(x for x in COMPARES if Path(fn).stem == x)
        _rec, rows = cmp_rows(scr / f"{cn}.json", cn, dev)
        return FITS[COMPARES[cn][0]][1], [rows[ds] for ds in WQ.EVAL_ORDER]

    pair, rcl = K.at(base, PAIR[c]), K.at(base, RECALL[c])
    pair.parent.mkdir(parents=True, exist_ok=True)
    with WQ.patch(SP, load=load):
        SP.pair([K.at(base, f"outputs/screen/{CMP[(sd, sp)]}.json") for sp in SPLITS], pair)
    null, sources = K.wqk_null(BASE, base, dev, SR.floor_of)
    for sp in SPLITS:
        cn = CMP[(sd, sp)]
        _r, kb = cmp_rows(scr / f"{cn}.json", cn, dev)
        null[sp] = {ds: {**v, "seed0": kb[ds]["base"][0]} for ds, v in null[sp].items()}
        sources[f"R2WS zrckp seed {sd} {sp}"] = f"outputs/screen/{cn}.json"
    with WQ.patch(SR, load_null=lambda _files: (null, sources)):
        rec = SR.pair([], Path(str(pair) + ".json"), rcl)
    restamp(pair, rcl, c)
    return rec


def tag(r):
    return f"{r['split']} {r['dataset']}"


def f4(x, signed=False):
    return "n/a" if x is None else K.fmt(x, signed)


def seed0_calls(base):
    """R2GW's filed call per read for its seed-0 zrckpb (beside, deciding nothing): {read: call}, or {} when the file is
    absent or not R2GW's zrckpb re-call."""
    p = Path(base) / SEED0_BESIDE
    try:
        rec = WQ.read_json(p)
        if (rec.get("stage"), rec.get("arm"), rec.get("against")) != (GW.STAGE, ARM, BASE):
            return {}
        return {tag(r): r["call"] for r in rec["rows"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def verdict(base=None, out=None):
    """R2WS's verdict by the rule from zrckpb's re-calls at seeds 1 and 2 against WQK's zrckp of the same seed under WQK's
    zrckp null (sections 3 and 4). Beside, deciding nothing: R2GW's seed-0 call per read, R@5 per read, the weaker form,
    R2's term per fit and seconds and memory; outputs/r2ws/screen-verdict."""
    import screen_recall as SR
    base = Path(base or WQ.ROOT)
    od = Path(out) if out else base / VERDICT
    problems = []
    try:
        wqk_ready(base)
    except SystemExit as e:
        problems.append(str(e))
    gates, gp = filed_gates(base)
    problems += gp
    devs = {}
    for sd in SEEDS:
        for sp in SPLITS:
            try:
                devs[(sd, sp)] = seed_split_device(sd, sp, base)
            except SystemExit as e:
                problems.append(str(e))
    problems += fit_problems(base, gates)
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"R2WS's fits ran on {len(keys)} devices: {sorted(keys)}")
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    for cn in COMPARES:
        try:
            cmp_rows(K.screen_dir(base) / f"{cn}.json", cn, dev)
        except SystemExit as e:
            problems.append(str(e))
    problems += column_problems(base, dev)
    problems += K.floors_problems(base, SR.floor_of)
    rc = {}
    if not problems:
        try:
            for c in CALLS:
                rc[c] = recall(base, dev, c)
        except (SystemExit, KeyError) as e:
            problems.append(WQ.clean(e, base))
    whole = sorted((sp, ds) for sp in SPLITS for ds in WQ.EVAL_ORDER)
    for c, r in rc.items():
        if sorted((x["split"], x["dataset"]) for x in r["rows"]) != whole:
            problems.append(f"the re-call of {c} against {BASE} (seed {CALLS[c]}) holds {len(r['rows'])} reads, not the "
                            f"twelve of {', '.join(SPLITS)}")
    ok = bool(rc) and not problems
    rows = {c: {tag(r): r for r in rc[c]["rows"]} for c in rc} if ok else {}
    gains = {CALLS[c]: [t for t, r in rows[c].items() if r["call"] == "GAIN"] for c in rows}
    losses = {CALLS[c]: [t for t, r in rows[c].items() if r["call"] == "LOSS"] for c in rows}
    reproduced = [t for t in gains.get(SEEDS[0], []) if all(t in gains[sd] for sd in SEEDS)] if ok else []
    lost = any(losses.values()) if ok else False
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if lost or not reproduced else "ADOPT_FOR_FULL_RUN")
    nxt = NEXT[v]
    wv = base / (K.VERDICT + ".json")
    calls = {t: {f"s{CALLS[c]}": rows[c][t]["call"] for c in rows} for t in (rows[next(iter(rows))] if rows else {})}
    delta = {t: {f"s{CALLS[c]}": rows[c][t]["delta"] for c in rows} for t in calls}
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": BASE,
           "seeds": list(SEEDS), "fits": {f"s{sd}": {sp: FIT[(sd, sp)] for sp in SPLITS} for sd in SEEDS},
           "against": {f"s{sd}": {sp: BASE_FIT[(sd, sp)] for sp in SPLITS} for sd in SEEDS},
           "floors": FLOORS + ".json", "null_kind": NULL_KIND, "device": dev, "gates": sorted(gates), "next_form": nxt,
           "gains": {f"s{sd}": g for sd, g in gains.items()}, "losses": {f"s{sd}": x for sd, x in losses.items()},
           "reproduced": reproduced, "problems": problems, "calls": calls, "delta_r5": delta, "naming": NAMING,
           "wqk_verdict": WQ.sha_file(wv) if wv.is_file() else None, "zwqk_sha256": WQ.sha_src(K.__file__),
           "zr2ws_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}

    def names(d):
        return "; ".join(f"seed {sd}: {', '.join(x) or 'none'}" for sd, x in d.items())
    md = [f"# R2WS, zrckpb under WQ at seeds {' and '.join(str(s) for s in SEEDS)} on Kaggle's T4: **{v}**", "",
          f"Rule: {RULE}.", "",
          f"- **zrckpb** ({WHAT}) against WQK's zrckp of the same seed on the T4, re-called under WQK's zrckp seed null. "
          f"GAIN by seed: {names(gains) if ok else 'n/a'}. LOSS by seed: {names(losses) if ok else 'n/a'}. GAIN at every "
          f"seed (the same read): {', '.join(reproduced) or 'none'}"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in section 6 before any number): {nxt}.", f"- Device: {json.dumps(dev)}.",
          f"- Gates: {', '.join(g[:12] for g in sorted(gates)) or 'none'}.", f"- Naming: {NAMING}."]
    beside = {}
    if v != "INCOMPLETE":
        fr, scr = K.fits_dir(base), K.screen_dir(base)
        zspk = {}
        for sp in SPLITS:
            _r, zr = K.cmp_rows(scr / f"{ZSPK_CMP[sp]}.json", ZSPK_CMP[sp], dev)
            zspk[sp] = {ds: zr[ds]["base"][0] for ds in WQ.EVAL_ORDER}
        s0 = seed0_calls(base)
        r5 = {}
        for t in calls:
            sp, ds = t.split(" ", 1)
            one = {c: rows[c][t] for c in rows}
            first = next(iter(one.values()))
            r5[t] = {"read": first["read"], "rrf": first["rrf"], "zspk": zspk[sp][ds], "floor": first["floor"],
                     "seed0_call": s0.get(t),
                     **{f"s{CALLS[c]}": {"zrckp": r["base"], "zrckpb": r["new"], "delta": r["delta"], "call": r["call"],
                                         "call_filed": r["call_filed"]} for c, r in one.items()}}
        with_gain = [sd for sd in SEEDS if gains[sd]]
        only_one = sorted(t for t in {x for g in gains.values() for x in g} if sum(t in gains[sd] for sd in SEEDS) == 1)
        term = {}
        for (sd, sp), n in FIT.items():
            t = ((WQ.read_json(fr / n / "screen.json").get(KEY) or {}).get("r2") or {}).get("term") or {}
            ep = t.get("epochs") or []
            term[n] = {"questions": t.get("questions"), "questions_with_term": t.get("questions_with_term"),
                       "share_with_term": t.get("share_with_term"),
                       "br_nll_first": ep[0].get("br_nll_mean") if ep else None,
                       "br_nll_last": ep[-1].get("br_nll_mean") if ep else None}
        tm = {FIT[k]: {"zrckpb": K.timing(FIT[k], fr, TIMING_FILE), "zrckp": K.timing(BASE_FIT[k], fr, "wqk_timing.json")}
              for k in FIT}
        rp = {c: Path(str(K.at(base, RECALL[c])) + ".json") for c in CALLS}
        beside = {"seed0_source": SEED0_BESIDE if s0 else None, "seed0_calls": s0, "seeds_with_a_gain": with_gain,
                  "gain_at_one_seed_only": only_one, "r5": r5, "terms": term, "timing": tm,
                  "pairs": {c: PAIR[c] + ".json" for c in CALLS}, "recalls": {c: RECALL[c] + ".json" for c in CALLS},
                  "recall_sha256": {c: WQ.sha_file(p) for c, p in rp.items()}}
        md += ["", "Beside, deciding nothing.", "",
               f"- Seeds with at least one GAIN: {with_gain or 'none'} of {list(SEEDS)} (the weaker form; it does not adopt). "
               f"Reads that are a GAIN at one seed only: {', '.join(only_one) or 'none'}.",
               f"- R2GW's seed-0 call per read ({'filed re-call ' + SEED0_BESIDE if s0 else 'n/a, not on this disk'}), "
               f"read before this file was written; it is not in the rule.", "",
               "R@5 per read on the T4 (zrckp and zrckpb of each seed, their difference and the call under the floor):", "",
               "| read | rrf | zspk | floor | " + " | ".join(f"zrckp s{sd} | zrckpb s{sd} | diff s{sd} | call s{sd}"
                                                           for sd in SEEDS) + " | seed 0 (R2GW) |",
               "|---|---:|---:|---:|" + "---:|---:|---:|---|" * len(SEEDS) + "---|"]
        for t, x in r5.items():
            md.append(f"| {t} ({x['read']}) | {f4(x['rrf'])} | {f4(x['zspk'])} | {f4(x['floor'])} | "
                      + " | ".join(f"{f4(x[f's{sd}']['zrckp'])} | {f4(x[f's{sd}']['zrckpb'])} | "
                                   f"{f4(x[f's{sd}']['delta'], True)} | {x[f's{sd}']['call']}" for sd in SEEDS)
                      + f" | {x['seed0_call'] or 'n/a'} |")
        md += ["", "R2's EM term on each fit (the share of training questions it reached, L_br at the first and the last "
                   "epoch):", "", "| fit | questions | with the term | share | L_br first | L_br last |",
               "|---|---:|---:|---:|---:|---:|"]
        for n, t in term.items():
            md.append(f"| {n} | {t['questions']} | {t['questions_with_term']} | {f4(t['share_with_term'])} "
                      f"| {f4(t['br_nll_first'])} | {f4(t['br_nll_last'])} |")

        def s_(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        md += ["", "Seconds and memory: zrckpb, and in brackets WQK's zrckp of the same seed and split, all on the T4:", "",
               "| fit | train s | read s | peak GPU GB, train / read | peak RAM GB |", "|---|---:|---:|---|---:|"]
        for n, t in tm.items():
            a_, b_ = t["zrckpb"], t["zrckp"]
            md.append(f"| {n} | {s_(a_['train_seconds'])} ({s_(b_['train_seconds'])}) | {s_(a_['read_seconds'])} "
                      f"({s_(b_['read_seconds'])}) | {f4(a_['train_peak_gpu_gb'])} / {f4(a_['read_peak_gpu_gb'])} "
                      f"({f4(b_['train_peak_gpu_gb'])} / {f4(b_['read_peak_gpu_gb'])}) | {f4(a_['peak_gb'])} "
                      f"({f4(b_['peak_gb'])}) |")
    rec["beside"] = beside
    WQ.write(od, rec, md)
    log(f"verdict: {v} (GAIN {names(gains) if ok else 'n/a'}; LOSS {names(losses) if ok else 'n/a'}; reproduced "
        f"{reproduced or 'none'}{'; ' + '; '.join(problems) if problems else ''}); next: {nxt}")
    return rec


def real_state():
    """What R2WS has under the real folders: its fits, its compares and everything under outputs/r2ws."""
    w = WQ.ROOT / NEW
    return (sorted(p.name for p in K.fits_dir(WQ.ROOT).glob("scr-r2ws-*")),
            sorted(p.name for p in K.screen_dir(WQ.ROOT).glob("scr-r2ws-*")),
            sorted((p.relative_to(w).as_posix(), p.stat().st_mtime_ns) for p in w.rglob("*")) if w.is_dir() else [])


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_fit(fits_root, name, gate_sha, overlay_sha, dev=None, read_=True, rows_cap=READ_CAP, r2=None, r2g=None,
             seed=None, wq_seed=None):
    """An R2WS fit as the kernel leaves it: screen.json (WQK's wq block, R2WS's block) and read.json with their stamps,
    train.json, r2ws_timing.json, a read and models.pt. seed and wq_seed put another seed in R2WS's and WQK's blocks."""
    sd, sp = FITS[name]
    d = Path(fits_root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    dev = dev or K.device_of(K.TOY_ENV)
    r2 = r2 or {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(R2_PARAMS),
                "term": {"lam": 1.0, "questions": 100, "questions_with_term": 90, "share_with_term": 0.9,
                         "epochs": [{"br_nll_mean": 4.6}, {"br_nll_mean": 3.9}]}}
    ks = sd if seed is None else seed
    k = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": ks, "split": sp, "device": dev, "gate": gate_sha,
         "overlay": overlay_sha, "r2": r2, "head": {"r2g": r2g}}
    WQ.write_json(d / "screen.json", {"arm": ARM, "seed": ks, "split": sp, "rc": 0, KEY: k,
                                      "wq": {"declared_in": K.DECLARED, "split": sp, "seed": sd if wq_seed is None else wq_seed,
                                             "arm": ARM, "gate": gate_sha}})
    WQ.write_json(d / "train.json", {"seconds": 150.0, "peak_gpu_gb": 3.5, "data_gb": 2.0})
    WQ.write_json(d / TIMING_FILE, {"peak_gb": 9.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read_:
        WQ.write_json(d / "read.json", {
            "seconds": 12.0, "peak_gpu_gb": 4.5, "carves": {c: {} for c in K.SIX},
            KEY: {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": ks, "split": sp, "device": dev,
                  "gate": gate_sha, "rows_cap": rows_cap}})
    return d


def selftest():
    import tempfile
    import types
    t0 = time.time()
    before = real_state()
    stops, TOY = K.stops, K.TOY_ENV
    me = sys.modules[__name__]
    R = GW.import_r2()
    R2 = R["R2"]
    # 1. the names: four fits (seeds 1 and 2 on both splits), four compares each with WQK's zrckp of the same seed, the
    #    seeds WQK's null seeds, the modules the stamps hash, R2WS's arm R2GW's zrckpb
    assert FIT == {(1, "WQ-musique"): "scr-r2ws-zrckpb-s1", (1, "WQ-hotpotqa"): "scr-r2ws-zrckpb-s1-hp",
                   (2, "WQ-musique"): "scr-r2ws-zrckpb-s2", (2, "WQ-hotpotqa"): "scr-r2ws-zrckpb-s2-hp"} and len(FITS) == 4
    assert SEEDS == tuple(WQ.NULL_SEEDS) == (1, 2) and ARM == "zrckpb" == GW.BESIDE_ARM and BASE == "zrckp"
    assert COMPARES["scr-r2ws-zrckpb-s2-vs-zrckp-s2-hp"] == ("scr-r2ws-zrckpb-s2-hp", ["scr-wqk-zrckp-s2-hp"])
    assert COMPARES["scr-r2ws-zrckpb-s1-vs-zrckp-s1"] == ("scr-r2ws-zrckpb-s1", ["scr-wqk-zrckp-s1"]) and len(COMPARES) == 4
    assert BASE_FIT[(2, "WQ-musique")] == "scr-wqk-zrckp-s2" and SEED0_FIT["WQ-hotpotqa"] == "scr-wqk-zrckp-s0-hp"
    assert BESIDE_FIT == {"WQ-musique": "scr-r2gw-zrckpb", "WQ-hotpotqa": "scr-r2gw-zrckpb-hp"}
    assert FITS["scr-r2ws-zrckpb-s2-hp"] == (2, "WQ-hotpotqa") and set(CALLS.values()) == set(SEEDS)
    assert all((HERE / f"{m}.py").is_file() for m in CODE) and len(set(CODE)) == len(CODE) and "zr2ws" in CODE
    assert "zlink" not in CODE and set(NEXT) == {"ADOPT_FOR_FULL_RUN", "NOT_ADOPTED", "INCOMPLETE"}
    assert NAMING.startswith(GW.NAMING_R2T) and "pending" in NEXT["ADOPT_FOR_FULL_RUN"] and "R2EW" in NEXT["NOT_ADOPTED"]
    assert R2.constants() == R2_CONST and R2.n_params() == R2_N == 59537 and R2.S.ARMS[ARM] == R2.ARMS[ARM]
    assert train_args("scr-r2ws-zrckpb-s2-hp") == ["train", "--name", "scr-r2ws-zrckpb-s2-hp", "--arm", "zrckpb", "--seed", "2",
                                                   "--device", "cuda", "--cache-root", K.ROOTS["cache"]]
    assert train_args("scr-r2ws-zrckpb-s1")[6] == "1"
    for bad in ("scr-r2gw-zrckpb", "scr-wqk-zrckp-s1", "scr-r2ws-zrckpb", "scr-r2ws-zrckpb-s0", "scr-r2ws-zrckpb-s3",
                "scr-r2ws-zrckpbe-s1", "scr-r2ew-zrckpbe"):
        stops(arm_seed_split, bad, has="not one of R2WS's fits")
        stops(train_args, bad, has="not one of R2WS's fits")
        stops(train, bad, TOY, has="not one of R2WS's fits")
    # 2. the guards: where, the read cap, WQK's verdict (train, read and fit refuse before reading any data)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = td / "repo"
        for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"}, {"image": "kaggle-gpu:other"}):
            stops(where_ok, {**TOY, **bad}, "train", has="R2WS trains in a Kaggle kernel")
            stops(train, FIT[(1, "WQ-musique")], {**TOY, **bad}, has="R2WS trains in a Kaggle kernel")
            stops(read, FIT[(2, "WQ-hotpotqa")], {**TOY, **bad}, has="R2WS reads in a Kaggle kernel")
        where_ok(TOY, "read")
        cap_ok(READ_CAP)
        for cap in (0, None, 32768):
            stops(cap_ok, cap, has="--rows-cap 65536")
            stops(read, FIT[(1, "WQ-musique")], TOY, rows_cap=cap, has="--rows-cap 65536")
            stops(fit, FIT[(1, "WQ-musique")], rows_cap=cap, has="--rows-cap 65536")
        stops(fit, FIT[(2, "WQ-hotpotqa")], has="R2WS trains in a Kaggle kernel")              # the laptop is not a T4
        stops(wqk_ready, B, has="base-verdict.json missing")
        wv = B / (K.VERDICT + ".json")
        with WQ.patch(WQ, ROOT=B):
            stops(train, FIT[(1, "WQ-musique")], TOY, has="base-verdict.json missing")
            for bad in ({"verdict": "INCOMPLETE"}, {"declared_in": WQ.DECLARED}, {"stage": "WQ"}):
                WQ.write_json(wv, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY", **bad})
                stops(wqk_ready, B, has="not WQK's BASE_READY")
                stops(train, FIT[(2, "WQ-hotpotqa")], TOY, has="not WQK's BASE_READY")
        WQ.write_json(wv, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        assert wqk_ready(B)["verdict"] == "BASE_READY"
        shutil.rmtree(B / "outputs")
        # 3. the seed reaches zp1c3p's train and the stamps: train() on a toy root, zp1c3p's train replaced by a recorder
        B7 = td / "stamp"
        for k_, r_ in (("GATE", {"verdict": "PASS"}), ("OVERLAY", {"status": "OK"}), ("CHECK", {"status": "REPRODUCED"})):
            WQ.write_json(B7 / K.RECORDS[k_], {"declared_in": K.DECLARED, **r_})
        WQ.write_json(B7 / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        summ = {"lam": 1.0, "batches": 10, "batches_with_term": 9, "questions": 100, "questions_with_term": 90,
                "share_with_term": 0.9, "epochs": [{"br_nll_mean": 4.6}, {"br_nll_mean": 3.9}]}
        fr7 = K.fits_dir(B7)
        import zp1c3p
        seen = []

        def rec_train(args, split):
            nm, sd_ = args[args.index("--name") + 1], int(args[args.index("--seed") + 1])
            seen.append((nm, args[args.index("--arm") + 1], sd_, split))
            WQ.write_json(WQ.fit_dir(nm, B7) / "screen.json", {"arm": ARM, "seed": sd_, "split": split, "rc": 0})
            return 0

        with WQ.patch(WQ, ROOT=B7, gate_passed=lambda base=None: None), WQ.patch(zp1c3p, train=rec_train), \
                WQ.patch(R2, summary=lambda: summ), WQ.patch(GW, fit_records=lambda split: {"kt_records": {"p": split}}):
            for (sd, sp), nm in FIT.items():
                assert train(nm, TOY) == 0
                s7, k7 = fit_stamp(nm, fr7)
                assert (k7["seed"], s7["seed"], s7["wq"]["seed"], k7["split"], k7["against"]) == (sd, sd, sd, sp, BASE_FIT[(sd, sp)])
        assert seen == [(nm, ARM, sd, sp) for (sd, sp), nm in FIT.items()], seen
        for (sd, sp), nm in FIT.items():
            s7, k7 = fit_stamp(nm, fr7)
            assert (k7["declared_in"], k7["device"], k7["seed0_base"], k7["beside"], k7["what"]) == (
                DECLARED, K.device_of(TOY), SEED0_FIT[sp], BESIDE_FIT[sp], WHAT)
            assert (s7["wq"]["declared_in"], s7["wq"]["arm"], s7["wq"]["gate"]) == (K.DECLARED, ARM, WQ.sha_file(B7 / K.RECORDS["GATE"]))
            assert k7["gate"] == s7["wq"]["gate"] and k7["wqk_verdict"] == WQ.sha_file(B7 / (K.VERDICT + ".json"))
            assert (k7["r2"]["constants"], k7["r2"]["params"], k7["r2"]["term"]) == (R2_CONST, list(R2_PARAMS), summ)
            assert k7["head"]["r2g"] is None and k7["head"]["read_rows"] == READ_CAP and k7["kt_records"] == {"p": sp}
            assert s7["zr2br_sha256"] == WQ.sha_src(HERE / "zr2br.py") and s7["zr2ws_sha256"] == WQ.sha_src(__file__)
            assert s7["zr2gw_sha256"] == WQ.sha_src(HERE / "zr2gw.py") and set(k7["code"]) == set(CODE)
        # a record of another seed is not the fit's: the stamp names the seed and fit_stamp refuses the mix
        nm1, nm2 = FIT[(1, "WQ-musique")], FIT[(2, "WQ-musique")]
        sj = WQ.fit_dir(nm2, B7) / "screen.json"
        keep = sj.read_bytes()
        s_ = WQ.read_json(sj)
        s_[KEY]["seed"] = 1
        WQ.write_json(sj, s_)
        stops(fit_stamp, nm2, fr7, has="at seed 2 (no R2WS stamp)")
        s_[KEY]["seed"], s_["seed"] = 2, 1
        WQ.write_json(sj, s_)
        stops(fit_stamp, nm2, fr7, has="at seed 2 (no R2WS stamp)")
        sj.write_bytes(keep)
        stops(fit_stamp, nm1, fr7 / "none", has="no R2WS stamp")
        # a train whose R2 term never ran stops unstamped
        with WQ.patch(WQ, ROOT=B7, gate_passed=lambda base=None: None), WQ.patch(zp1c3p, train=rec_train), \
                WQ.patch(R2, summary=lambda: {**summ, "batches_with_term": 0}):
            stops(train, nm1, TOY, has="carried R2's term")
        # 4. the read's stamp, fit_device, and the read's refusals on the stamped fits
        n7 = FIT[(2, "WQ-hotpotqa")]
        WQ.write_json(WQ.fit_dir(n7, B7) / "read.json", {"seconds": 12.0, "carves": {c: {} for c in K.SIX}})
        stops(fit_device, n7, fr7, has="has no R2WS read on the device it was fitted on")
        r7 = stamp_read(n7, TOY, {"webqsp/fit": "k"}, READ_CAP, B7)[KEY]
        assert (r7["seed"], r7["split"], r7["rows_cap"], r7["kt_records"]) == (2, "WQ-hotpotqa", READ_CAP, {"webqsp/fit": "k"})
        assert fit_device(n7, fr7) == K.device_of(TOY)
        stamp_read(n7, TOY, {"webqsp/fit": "k"}, 32768, B7)
        stops(fit_device, n7, fr7, has="has no R2WS read on the device it was fitted on")
        stamp_read(n7, TOY, {"webqsp/fit": "k"}, READ_CAP, B7)
        rj = WQ.fit_dir(n7, B7) / "read.json"
        assert stamped(rj, n7) and not stamped(rj, nm1)
        boom = types.SimpleNamespace(n=0)

        def gate_boom(base=None):
            boom.n += 1
            raise SystemExit("zwq: outputs/wqk/gate.json is FAIL, not WQ's PASS")

        with WQ.patch(WQ, fit_dir=lambda name, base=B7: Path(base) / "outputs" / "screen" / "fits" / name):
            stops(read, n7, {**TOY, "cudnn": 1}, has="this read runs on")
            sj7 = WQ.fit_dir(n7, B7) / "screen.json"
            keep7 = sj7.read_bytes()
            s7 = WQ.read_json(sj7)
            s7[KEY]["r2"]["constants"] = {**R2_CONST, "K": 6}
            WQ.write_json(sj7, s7)
            stops(read, n7, TOY, has="fitted with R2 constants")
            sj7.write_bytes(keep7)
            with WQ.patch(WQ, gate_passed=gate_boom):
                stops(read, n7, TOY, has="is FAIL")
            assert boom.n == 1
        # 5. take, compare-all and the verdict on a toy of WQK's base set (zwqk's toy, its fits, compares and floors) and an
        #    R2WS session: ADOPT_FOR_FULL_RUN, NOT_ADOPTED each way, and INCOMPLETE for each missing or wrong piece
        K.toy_build(B)
        gw, ow = K.fake_session(td / "wqk_out", B, tag="w")
        for nm in K.FITS:
            K.fake_fit(K.fits_dir(B), nm, gw, ow)

        def wqk_run(new, base_fit, out):                # zwqk's selftest's: WQK's compares on the toy
            new, b0, c_ = Path(new).name, Path(base_fit).name, Path(out).name
            arm, seed, sp = K.FITS[new]
            dd = {ds: (0.002, -0.002)[seed - 1] for ds in WQ.EVAL_ORDER} if arm != "zg6" else {}
            if (arm, sp) == ("zrckp", "WQ-musique"):
                dd["musique"] = (0.02, -0.01)[seed - 1]
            K.fake_compare(B, c_, new, b0, dd, base_r5=0.59 if arm == "zrckp" else 0.6)
            return 0

        why = WQ.wq_why
        with WQ.patch(WQ, run_compare=wqk_run, wq_why=lambda name, base=B: why(name, base)):
            assert K.compare_all(B) == 0
        assert K.floors(B) == 0
        WQ.write_json(B / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        out_s = td / "r2ws_out"
        gs, os_ = K.fake_session(out_s, B, tag="s")
        names = list(FITS)
        for nm in names[:-1]:
            fake_fit(out_s / "fits", nm, gs, os_)
        fake_fit(out_s / "fits", names[-1], gs, os_, read_=False)
        took, left = take(out_s, B)
        assert len(took) == 3 * 5 + 2 and len(left) == 1 and "has no R2WS read" in left[0], (took, left)
        fr = K.fits_dir(B)
        assert not (fr / names[0] / "models.pt").exists() and (fr / names[0] / TIMING_FILE).is_file()
        assert (B / SESSIONS / gs[:12] / "gate.json").is_file() and not (fr / names[-1]).exists()
        fake_fit(out_s / "fits", names[-1], gs, os_)
        took, left = take(out_s, B)
        assert len(took) == 5 and not left and take(out_s, B) == ([], [])
        (fr / names[0] / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out_s, B, has="exists with other bytes")
        shutil.copyfile(out_s / "fits" / names[0] / "train.json", fr / names[0] / "train.json")
        for other in (BASE_FIT[(1, "WQ-musique")], BESIDE_FIT["WQ-musique"]):         # WQK's and R2GW's fits are theirs to take
            (out_s / "fits" / other).mkdir()
            stops(take, out_s, B, has="are not R2WS's fits")
            (out_s / "fits" / other).rmdir()
        stops(take, td / "wqk_out" / "wqk", B, has="holds no wqk/gate.json")
        cur = {}

        def r2ws_run(new, base_fit, out):               # lean_screen's compare with one base, in its record's form
            import lean_screen as S
            n_, c_, b_ = Path(new).name, Path(out).name, Path(base_fit).name
            sd_, sp = FITS[n_]
            assert [b_] == COMPARES[c_][1] and c_ == CMP[(sd_, sp)], (b_, c_)
            wrows = K.cmp_rows(K.screen_dir(B) / f"{b_}.json", b_)[1]                   # WQK's read of that seed
            rows = []
            for ds in WQ.EVAL_ORDER:
                r5 = wrows[ds]["new"][0]
                nw = round(r5 + cur.get((sd_, sp, ds), 0.0), 4)
                dl = round(nw - r5, 4)
                lo, hi = round(dl - 0.004, 4), round(dl + 0.004, 4)
                rows.append({"dataset": ds, "read": "in-domain" if ds in WQ.SPLITS[sp] else "zero-shot", "questions": 100,
                             "base": [r5, 0.5, 0.4], "new": [nw, 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                             "delta": [dl, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(dl, lo, hi)})
            bp = f"outputs/screen/fits/{b_}"
            WQ.write_json(K.screen_dir(B) / f"{c_}.json", {
                "new": f"outputs/screen/fits/{n_}", "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
                "trained_on": sorted(WQ.SPLITS[sp]), "by_base": {bp: {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]),
                                                                      "decides": True}},
                "verdict": S.verdict([r["call"] for r in rows])})
            return 0

        def run(deltas):
            cur.clear()
            cur.update(deltas)
            for c_ in COMPARES:
                (K.screen_dir(B) / f"{c_}.json").unlink(missing_ok=True)
            with WQ.patch(WQ, run_compare=r2ws_run):
                assert compare_all(B) == 0 and compare_all(B) == 0
            return verdict(B)

        HS, MW, MM = "WQ-hotpotqa squad", "WQ-musique webqsp", "WQ-musique musique"
        v = run({(1, "WQ-hotpotqa", "squad"): 0.03, (2, "WQ-hotpotqa", "squad"): 0.03, (1, "WQ-musique", "webqsp"): 0.02})
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and not v["problems"], v["problems"]
        assert sorted(v["gains"]["s1"]) == sorted([HS, MW]) and v["gains"]["s2"] == [HS] and v["reproduced"] == [HS]
        assert v["losses"] == {"s1": [], "s2": []} and v["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["gates"] == [gs] and v["device"] == K.device_of(TOY) and len(v["calls"]) == 12 and v["naming"] == NAMING
        assert v["calls"][HS] == {"s1": "GAIN", "s2": "GAIN"} and v["calls"][MW] == {"s1": "GAIN", "s2": "WITHIN"}
        assert v["fits"]["s2"] == {"WQ-musique": "scr-r2ws-zrckpb-s2", "WQ-hotpotqa": "scr-r2ws-zrckpb-s2-hp"}
        assert v["against"]["s1"]["WQ-hotpotqa"] == "scr-wqk-zrckp-s1-hp" and v["seeds"] == [1, 2]
        k_ = WQ.read_json(K.screen_dir(B) / f"{CMP[(2, 'WQ-hotpotqa')]}.json")[KEY]
        assert (k_["new"], k_["bases"], k_["device"], k_["seed"]) == (FIT[(2, "WQ-hotpotqa")], ["scr-wqk-zrckp-s2-hp"],
                                                                      K.device_of(TOY), 2)
        b = v["beside"]
        assert b["seeds_with_a_gain"] == [1, 2] and b["gain_at_one_seed_only"] == [MW] and b["seed0_calls"] == {}
        x = b["r5"][HS]
        assert (x["read"], x["zspk"], x["floor"], x["seed0_call"]) == ("in-domain", 0.6, 0.0075, None)
        assert x["s1"] == {"zrckp": 0.592, "zrckpb": 0.622, "delta": 0.03, "call": "GAIN", "call_filed": "GAIN"}, x["s1"]
        assert x["s2"]["zrckp"] == 0.588 and x["s2"]["zrckpb"] == 0.618 and x["s2"]["call"] == "GAIN"
        assert b["r5"][MM]["floor"] == round(2 * (((0.02 ** 2 + 0.01 ** 2) / 2) ** 0.5), 6)
        assert b["terms"][FIT[(1, "WQ-musique")]] == {"questions": 100, "questions_with_term": 90, "share_with_term": 0.9,
                                                      "br_nll_first": 4.6, "br_nll_last": 3.9}
        assert b["timing"][FIT[(2, "WQ-musique")]]["zrckpb"] == {"train_seconds": 150.0, "read_seconds": 12.0,
                                                                  "train_peak_gpu_gb": 3.5, "read_peak_gpu_gb": 4.5,
                                                                  "data_gb": 2.0, "peak_gb": 9.0}
        assert b["timing"][FIT[(1, "WQ-hotpotqa")]]["zrckp"]["peak_gb"] == 8.0
        for c_ in CALLS:
            assert b["recall_sha256"][c_] == WQ.sha_file(B / (RECALL[c_] + ".json"))
            r_ = WQ.read_json(B / (RECALL[c_] + ".json"))
            assert (r_["stage"], r_["seed"], r_["against"], r_["deciding"], len(r_["rows"])) == (
                STAGE, CALLS[c_], f"{BASE} (seed {CALLS[c_]})", True, 12)
            assert (B / (PAIR[c_] + ".md")).read_text(encoding="utf-8").startswith(f"R2WS ({DECLARED})")
        r1 = WQ.read_json(B / (RECALL["zrckpb-s1"] + ".json"))
        assert {(r["split"], r["dataset"]): r["base"] for r in r1["rows"]}[("WQ-hotpotqa", "squad")] == 0.592   # seed 1's zrckp
        r2_ = WQ.read_json(B / (RECALL["zrckpb-s2"] + ".json"))
        assert {(r["split"], r["dataset"]): r["base"] for r in r2_["rows"]}[("WQ-hotpotqa", "squad")] == 0.588  # seed 2's zrckp
        md = (B / (VERDICT + ".md")).read_text(encoding="utf-8")
        assert "**ADOPT_FOR_FULL_RUN**" in md and "| WQ-hotpotqa squad (in-domain) | 0.4000 | 0.6000 | 0.0075 |" in md
        assert "| 0.5920 | 0.6220 | +0.0300 | GAIN | 0.5880 | 0.6180 | +0.0300 | GAIN | n/a |" in md
        # R2GW's seed-0 call per read is read from its filed re-call (beside; not in the rule)
        WQ.write_json(B / SEED0_BESIDE, {"stage": GW.STAGE, "arm": ARM, "against": BASE,
                                         "rows": [{"split": "WQ-hotpotqa", "dataset": "squad", "call": "WITHIN"},
                                                  {"split": "WQ-musique", "dataset": "webqsp", "call": "GAIN"}]})
        v = verdict(B)
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and v["beside"]["seed0_calls"] == {HS: "WITHIN", MW: "GAIN"}
        assert v["beside"]["r5"][HS]["seed0_call"] == "WITHIN" and v["beside"]["r5"][MW]["seed0_call"] == "GAIN"
        WQ.write_json(B / SEED0_BESIDE, {"stage": GW.STAGE, "arm": "zrckpbg", "against": BASE, "rows": []})
        assert verdict(B)["beside"]["seed0_calls"] == {}
        # a LOSS at either seed, a seed with no GAIN, GAINs on different reads, and a GAIN within a raised floor: NOT_ADOPTED
        v = run({(1, "WQ-hotpotqa", "squad"): 0.03, (2, "WQ-hotpotqa", "squad"): 0.03, (2, "WQ-musique", "webqsp"): -0.03})
        assert (v["verdict"], v["losses"], v["reproduced"], v["next_form"]) == (
            "NOT_ADOPTED", {"s1": [], "s2": [MW]}, [HS], NEXT["NOT_ADOPTED"])
        v = run({(1, "WQ-hotpotqa", "squad"): 0.03, (2, "WQ-hotpotqa", "squad"): 0.03, (1, "WQ-hotpotqa", "2wiki"): -0.03})
        assert (v["verdict"], v["losses"]["s1"]) == ("NOT_ADOPTED", ["WQ-hotpotqa 2wiki"])
        v = run({(1, "WQ-hotpotqa", "squad"): 0.03})
        assert (v["verdict"], v["gains"], v["reproduced"]) == ("NOT_ADOPTED", {"s1": [HS], "s2": []}, [])
        assert v["beside"]["seeds_with_a_gain"] == [1]
        v = run({(1, "WQ-hotpotqa", "squad"): 0.03, (2, "WQ-hotpotqa", "2wiki"): 0.03})       # a GAIN each seed, not the same read
        assert (v["verdict"], v["reproduced"], v["losses"]) == ("NOT_ADOPTED", [], {"s1": [], "s2": []})
        assert v["beside"]["seeds_with_a_gain"] == [1, 2] and v["beside"]["gain_at_one_seed_only"] == [
            "WQ-hotpotqa 2wiki", HS] and v["next_form"] == NEXT["NOT_ADOPTED"]
        v = run({(1, "WQ-musique", "musique"): 0.02, (2, "WQ-musique", "musique"): 0.02})     # filed GAINs within the raised floor
        assert (v["verdict"], v["gains"], v["losses"]) == ("NOT_ADOPTED", {"s1": [], "s2": []}, {"s1": [], "s2": []})
        assert v["calls"][MM] == {"s1": "WITHIN", "s2": "WITHIN"} and v["delta_r5"][MM] == {"s1": 0.02, "s2": 0.02}
        r0 = WQ.read_json(B / (RECALL["zrckpb-s1"] + ".json"))
        assert [r["call_filed"] for r in r0["rows"] if (r["split"], r["dataset"]) == ("WQ-musique", "musique")] == ["GAIN"]
        v = run({(1, "WQ-hotpotqa", "squad"): 0.03, (2, "WQ-hotpotqa", "squad"): 0.03})
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN"

        def incomplete(has):
            r = verdict(B)
            assert r["verdict"] == "INCOMPLETE" and r["beside"] == {} and r["next_form"] == NEXT["INCOMPLETE"] and \
                any(has in p for p in r["problems"]), (has, r["problems"])

        def away(p, has):
            keep_ = p.read_bytes()
            p.unlink()
            incomplete(has)
            p.write_bytes(keep_)

        ce = K.screen_dir(B) / f"{CMP[(2, 'WQ-hotpotqa')]}.json"
        away(ce, f"{CMP[(2, 'WQ-hotpotqa')]}.json missing")
        away(B / (K.VERDICT + ".json"), "base-verdict.json missing")
        keep = ce.read_bytes()
        rc_ = WQ.read_json(ce)
        rc_[KEY]["new"] = "scr-r2ws-zrckpb-s1-hp"
        WQ.write_json(ce, rc_)
        incomplete("is not R2WS's compare of")
        rc_[KEY].update({"new": FIT[(2, "WQ-hotpotqa")], "seed": 1})
        WQ.write_json(ce, rc_)
        incomplete("is not R2WS's compare of")                                                  # another seed's stamp
        ce.write_bytes(keep)
        rc_ = WQ.read_json(ce)
        rc_["by_base"][rc_["bases"][0]]["rows"][0]["base"][0] = 0.5                              # not WQK's read of that seed
        WQ.write_json(ce, rc_)
        incomplete(f"{CMP[(2, 'WQ-hotpotqa')]}: its zrckp column is not WQK's read of scr-wqk-zrckp-s2-hp")
        ce.write_bytes(keep)
        n0, n1 = FIT[(1, "WQ-musique")], FIT[(2, "WQ-hotpotqa")]
        bad_r2 = {"declared_in": R2_DECLARED, "constants": {**R2_CONST, "K": 6}, "params": list(R2_PARAMS),
                  "term": {"lam": 1.0, "questions_with_term": 90}}
        no_term = {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(R2_PARAMS),
                   "term": {"lam": 1.0, "questions_with_term": 0}}
        for nm, kw, has in (
                (n0, {"dev": {**K.device_of(TOY), "gpu": "Tesla P100"}}, "ran on different devices"),
                (n1, {"gate_sha": "f" * 64}, f"{n1}: its train names the gate ffffffffffff, not a filed R2WS PASS gate"),
                (n0, {"r2": bad_r2}, f"{n0}: R2's constants or EM term are not as declared"),
                (n1, {"r2": no_term}, f"{n1}: R2's constants or EM term are not as declared"),
                (n0, {"rows_cap": 32768}, "has no R2WS read on the device it was fitted on"),
                (n1, {"r2g": GW.R2G}, f"{n1}: zrckpb is gated"),
                (n1, {"seed": 1}, "at seed 2 (no R2WS stamp)"),
                (n0, {"wq_seed": 2}, f"{n0}: its record names seeds (1, 2, 1, 1), not 1")):
            fake_fit(fr, nm, **{"gate_sha": gs, "overlay_sha": os_, **kw})
            incomplete(has)
            fake_fit(fr, nm, gs, os_)
        bk = td / "fit_backup"
        shutil.copytree(fr / n0, bk)
        shutil.rmtree(fr / n0)
        incomplete("no R2WS stamp")                                                              # a fit not taken
        shutil.copytree(bk, fr / n0)
        assert verdict(B)["verdict"] == "ADOPT_FOR_FULL_RUN"
        for p in (fr / BASE_FIT[(2, "WQ-hotpotqa")] / "read.json",):                            # WQK's zrckp fit of seed 2 unread
            away(p, "scr-wqk-zrckp-s2-hp has no WQK read on the device it was fitted on")
        out2 = td / "r2ws_out2"
        g2_, _o2 = K.fake_session(out2, B, verdict_="FAIL", tag="h")
        take(out2, B)
        incomplete("FAIL, not WQK's PASS")
        shutil.rmtree(B / SESSIONS / g2_[:12])
        p = K.screen_dir(B) / "scr-wqk-zrckp-s1.json"
        keep = p.read_bytes()
        r = WQ.read_json(p)
        r["by_base"]["outputs/screen/fits/scr-wqk-zrckp-s0"]["rows"][2]["delta"][0] = 0.05
        WQ.write_json(p, r)
        incomplete("floors-zrckp.json is not its compares' floors")
        p.write_bytes(keep)

        def short(base, dev, c_):
            rec = recall.__wrapped__(base, dev, c_)
            rec["rows"] = rec["rows"][:-1]
            return rec

        short_fn = types.FunctionType(short.__code__, {**globals(), "recall": types.SimpleNamespace(__wrapped__=recall)})
        with WQ.patch(me, recall=short_fn):
            incomplete("holds 11 reads, not the twelve")
        assert verdict(B)["verdict"] == "ADOPT_FOR_FULL_RUN"
    # 6. nothing left patched in zr2br or the carve, and nothing written under R2WS's real folders
    assert R2.STASH["S"] is None
    assert real_state() == before, (real_state(), before)
    print(f"zr2ws selftest: ok ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog="zr2ws.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=READ_CAP)
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
