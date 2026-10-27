"""WQ (docs/WQ_WEBQSP_TRAINS.md): webqsp trains and metaqa is read zero-shot. The lean pair (zrckp, zp1c3p.py; zspk,
zp1c3.py) is fitted on two splits that hold webqsp's fit carve (1,202 questions) beside three wiki training sets, the
wiki side left out one at a time as before, and read on the six s1eval carves at p@swa:
    WQ-musique    trains on webqsp fit, squad fit, hotpotqa fit, 2wiki fit; zero-shot musique, metaqa
    WQ-hotpotqa   trains on webqsp fit, squad fit, musique s1fit, 2wiki fit; zero-shot hotpotqa, metaqa
webqsp's fit carve goes through the stages that built P1c2's carves, each committed script imported unchanged and patched
in this process only, and everything it writes is under outputs/wq. The other ten carves are P1c2's (and P1c3's KT
input), placed at WQ's roots byte for byte. Nothing outside outputs/wq is written but the fits' own records and their
comparisons under outputs/screen.

    python outputs/mp_unified/zwq.py copy-bases                                 (W1: outputs/wq/bases.json)
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zwq.py s4d [--check] --threads 3 --host       (W3)
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zwq.py pools [--check] --threads 3 --host     (W4)
    python outputs/mp_unified/zwq.py check                                      (section 4: outputs/wq/check/check.json)
    python outputs/mp_unified/zwq.py look --shard i/3 --host                    (W5)
    python outputs/mp_unified/zwq.py build --what rmatch|zrc|links --host       (W7)
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zwq.py kt --threads 3 --host                  (W8)
    python outputs/mp_unified/zwq.py overlay                                    (W9: outputs/wq/overlay.json)
    python outputs/mp_unified/zwq.py gate                                       (W10: outputs/wq/gate.json)
    python outputs/mp_unified/zwq.py train --split WQ-musique --name scr-wq-zrckp-s0 --arm zrckp --seed 0 \\
        --device cuda --host --cache-root outputs/wq/p1c2/cache
    python outputs/mp_unified/zwq.py read --name scr-wq-zrckp-s0 --device cuda --host --cache-root outputs/wq/p1c2/cache
    python outputs/mp_unified/zwq.py compare --new outputs/screen/fits/scr-wq-zrckp-s1 \\
        --base outputs/screen/fits/scr-wq-zrckp-s0 --out outputs/screen/scr-wq-zrckp-s1
    python outputs/mp_unified/zwq.py floors --arm zrckp                         (outputs/wq/floors-zrckp.json, .md)
    python outputs/mp_unified/zwq.py beside --arm zrckp --split WQ-musique      (outputs/wq/beside/, decides nothing)
    python outputs/mp_unified/zwq.py summary                                    (outputs/wq/summary.json, .md)
    python outputs/mp_unified/zwq.py --selftest

W0, W2 and W6 are look_x_six.py's and lean_cache.py's own commands with WQ's roots. This file sets no thread variable:
each stage's modules set theirs as in the runs that built P1c2's carves (a look imports look_p1c2 before numpy loads; the
walks run under pylib_run, which loads numba first). check, overlay, gate, floors and summary exit 0 when their record
says OK (REPRODUCED, PASS, BASE_READY) and 2 after writing any other record.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DECLARED = "docs/WQ_WEBQSP_TRAINS.md"
STAGE = "WQ"
# WQ's roots, each written out whole so that a queued item naming this file keeps them from the host's clean-up
ROOTS = {"look": "outputs/wq/look", "s1cache": "outputs/wq/s1cache", "s4d": "outputs/wq/step4d/pools",
         "pools": "outputs/wq/p1c2/pools", "plook": "outputs/wq/p1c2/look", "cache": "outputs/wq/p1c2/cache",
         "rmatch": "outputs/wq/p1c2/chains/rmatch", "zrc": "outputs/wq/p1c2/chains/zrc", "links": "outputs/wq/p1c2/links",
         "kt": "outputs/wq/p1c3/kt"}
CHECK_ROOTS = {"s4d": "outputs/wq/check/step4d/pools", "pools": "outputs/wq/check/p1c2/pools"}
CHECK = "outputs/wq/check/check.json"
BASES_REC = "outputs/wq/bases.json"
OVERLAY = "outputs/wq/overlay.json"
GATE = "outputs/wq/gate.json"
SUMMARY = "outputs/wq/summary"
BESIDE = "outputs/wq/beside"
FLOORS = {"zrckp": "outputs/wq/floors-zrckp", "zspk": "outputs/wq/floors-zspk"}
SRC = {"cache": "outputs/p1c2/cache", "rmatch": "outputs/p1c2/chains/rmatch", "zrc": "outputs/p1c2/chains/zrc",
       "links": "outputs/p1c2/links", "kt": "outputs/p1c3/kt"}
STEP1_CACHE = "outputs/step1/cache"
FILED = {"step4d": ("outputs/step4d/pools/webqsp__s1eval.npz", "outputs/step4d/pools/pools.json"),
         "p1c2": ("outputs/p1c2/pools/webqsp/webqsp__s1eval.npz", "outputs/p1c2/pools/webqsp/pools.json")}
STEP4B_EVAL_SHA = "053cac56c16b6f7509114ebfcde60ab9b128e45e1764accc487ef3096115de1b"   # outputs/step4b/eval/webqsp.npz
FIT_IDS_SHA = "fecf94bbfdff25925abd771421d4ad7276889d81b8449891783506bd087bdb90"       # section 2: webqsp's fit carve
FIT_N = 1202
CONSTRUCTION = "KIND_T"
KT_DECLARED = "docs/P1C3_KT_INPUT.md"
SPLITS = {"WQ-musique": ("webqsp", "squad", "hotpotqa", "2wiki"), "WQ-hotpotqa": ("webqsp", "squad", "musique", "2wiki")}
SUF = {"WQ-musique": "", "WQ-hotpotqa": "-hp"}
ZERO_SHOT = {"WQ-musique": ("musique", "metaqa"), "WQ-hotpotqa": ("hotpotqa", "metaqa")}
FIT_CARVE = {"webqsp": "fit", "squad": "fit", "musique": "s1fit", "hotpotqa": "fit", "2wiki": "fit"}
EVAL_ORDER = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")      # lean_gpu's, checked on use
NEW = ("webqsp", "fit")
SCREEN = tuple((d, FIT_CARVE[d]) for d in FIT_CARVE) + tuple((d, "s1eval") for d in EVAL_ORDER)
OVERLAID = tuple(c for c in SCREEN if c != NEW)
CHAINED = (("webqsp", "fit"), ("webqsp", "s1eval"), ("metaqa", "s1eval"))
KINDS = ("cache", "rmatch", "zrc", "links", "kt")
BASES = ("2wiki", "hotpotqa")
ARMS = ("zrckp", "zspk")
SEEDS = (0, 1, 2)
NULL_SEEDS = (1, 2)
OLD_CARD = {"zrckp": "scr-p1c3p-zrckp", "zspk": "scr-p1c3-zspk"}
STAMP = {"zrckp": "p1c3p", "zspk": "p1c3"}
FLOOR = 0.0075                                                                  # lean_screen's, checked on use
for _t in (tuple(SPLITS), EVAL_ORDER, SCREEN, OVERLAID, CHAINED, KINDS, BASES, ARMS, SEEDS, tuple(FIT_CARVE)):
    assert len(set(_t)) == len(_t), _t
assert len(SCREEN) == 11 and len(OVERLAID) == 10 and set(CHAINED) <= set(SCREEN)
assert all(sorted(SPLITS[sp] + ZERO_SHOT[sp]) == sorted(EVAL_ORDER) for sp in SPLITS)
assert all(set(SPLITS[sp]) <= set(FIT_CARVE) for sp in SPLITS)


def fit_name(arm, seed, split):
    return f"scr-wq-{arm}-s{seed}{SUF[split]}"


FITS = {fit_name(a, sd, sp): (a, sd, sp) for a in ARMS for sd in SEEDS for sp in SPLITS}
assert len(FITS) == 12


def kinds_of(carve):
    """The folders a carve needs at WQ's roots: its cache, links and KT input; the KB carves also their chains."""
    return ("cache", "links", "kt") + (("rmatch", "zrc") if tuple(carve) in CHAINED else ())


# ── small helpers ────────────────────────────────────────────────────────────


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zwq: {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def write(out, rec, md):
    """out.json and out.md (screen_recall.write's pair, without its imports)."""
    write_json(Path(str(out) + ".json"), rec)
    Path(str(out) + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parts_of(p):
    return tuple(x for x in str(p).replace("\\", "/").split("/") if x)


def rel(p, base=ROOT):
    try:
        return Path(p).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(p).name


def clean(msg, base=ROOT):
    """A message without the root's absolute path or any home folder (records name repository paths only)."""
    s = str(msg)
    b = Path(base)
    for r in sorted({str(b), b.as_posix(), str(b.resolve()), b.resolve().as_posix()}, key=len, reverse=True):
        s = s.replace(r + "\\", "").replace(r + "/", "").replace(r, "")
    return re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s'\"]+", "<home>", s)


def must_stop(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not stop")


@contextlib.contextmanager
def patch(obj, **names):
    """obj's attributes set to names' values for the block, every one restored after (on a failure too)."""
    missing = [k for k in names if not hasattr(obj, k)]
    if missing:
        raise SystemExit(f"zwq: {getattr(obj, '__name__', obj)} has no {missing}")
    saved = {k: getattr(obj, k) for k in names}
    try:
        for k, v in names.items():
            setattr(obj, k, v)
        yield obj
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


def place(src, dst, sha, base=ROOT):
    """dst as src's bytes (src's sha256 is sha): a hard link, else a copy renamed into place. A file already at dst must
    be those bytes ("present"), or the stage stops; nothing is ever overwritten."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        if sha_file(dst) != sha:
            raise SystemExit(f"zwq: {rel(dst, base)} is present and is not {rel(src, base)}'s bytes")
        return "present"
    if sha_file(src) != sha:
        raise SystemExit(f"zwq: {rel(src, base)} is not the sha256 {sha[:12]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except FileExistsError:
        return place(src, dst, sha, base)
    except OSError:
        tmp = dst.with_name(f"{dst.name}.wq{os.getpid()}.tmp")
        shutil.copyfile(src, tmp)
        try:
            os.rename(tmp, dst)                         # Windows refuses a rename over a file
        except FileExistsError:
            tmp.unlink()
            return place(src, dst, sha, base)
        if sha_file(dst) != sha:
            raise SystemExit(f"zwq: {rel(dst, base)} was copied, but its sha256 is not {sha[:12]}")
        return "copied"
    if not os.path.samefile(src, dst):
        raise SystemExit(f"zwq: {rel(dst, base)} was linked, but it is not {rel(src, base)}")
    return "linked"


def brief(chk):
    return (f"units {chk['units']}, missing {len(chk['missing'])}, differs {len(chk['differs'])}, no record "
            f"{chk['no_record']}, short {chk['short']}")


# ── the pool stages (W3, W4, W8 and the check) ────────────────────────────────


def with_fit(base):
    """base (step 4's Opened, or the mirror's) with webqsp's fit carve beside carves.json's (section 2: the carve W0's
    look read, m3b_compile's fit population): step 4c's populations then build and check it as any other fit carve."""

    class OpenedWQ(base):
        def __init__(self):
            super().__init__()
            pd = self.carves["per_dataset"]["webqsp"]
            ent = {"name": "fit", "n": FIT_N, "sha256": FIT_IDS_SHA}
            if pd.get("fit", ent) != ent:
                raise SystemExit(f"zwq: carves.json holds a webqsp fit carve {pd['fit']}, not W0's")
            pd["fit"] = ent

    OpenedWQ.__name__ = OpenedWQ.__qualname__ = f"WQ{base.__name__}"
    return OpenedWQ


@contextlib.contextmanager
def on_wq_pools(check=False, host=False):
    """Step 4d's, P1c2's and P1c3's stages on webqsp's fit carve, written under outputs/wq; with check, on its s1eval carve
    (their filed run) written under outputs/wq/check. P1e's host mode (the mirror) as P1c2's pools ran it."""
    import p1c2_pools as PC2                    # numba-free at import
    import p1e_anchor_kind as K                 # P1e's walks (imports numba)
    import step4b_pool_reach as S4B
    E, S4 = K.E, K.S4
    C4, D4 = E.C4, E.D4
    if PC2.C4S is not C4 or E.S4 is not S4 or S4B.S4 is not S4:
        raise SystemExit("zwq: the pool stages do not share step 4c's and step 4's modules")
    if host:
        K.G.on_the_mirror()
    ns = types.SimpleNamespace(PC2=PC2, K=K, E=E, S4=S4, C4=C4, D4=D4)
    s4d = ROOT / (CHECK_ROOTS if check else ROOTS)["s4d"]
    with contextlib.ExitStack() as st:
        st.enter_context(patch(S4, Opened=with_fit(S4.Opened)))
        st.enter_context(patch(D4, POOLS=s4d, MANIFEST=s4d / "pools.json"))
        if check:
            st.enter_context(patch(PC2, POOLS=ROOT / CHECK_ROOTS["pools"]))
        else:
            orig_of, orig_for = C4.carves_of, PC2.carves_for
            st.enter_context(patch(C4, carves_of=lambda name: ("fit",) if name == "webqsp" else orig_of(name),
                                   STEP1_CACHE=ROOT / ROOTS["s1cache"]))
            st.enter_context(patch(PC2, carves_for=lambda name: ["fit"] if name == "webqsp" else orig_for(name),
                                   POOLS=ROOT / ROOTS["pools"], CACHE=ROOT / ROOTS["cache"]))
        yield ns


def pool_entry_problems(ent, carve, d, kind, base=ROOT):
    """Why a pools entry (step 4d's: kind step4d; P1c2's: kind p1c2) of webqsp/carve, its file under d, is not the one
    WQ declares, or []."""
    if ent is None:
        return [f"{kind}: no webqsp/{carve} entry"]
    out = []
    p = Path(d) / str(ent.get("file"))
    if not p.is_file() or sha_file(p) != ent.get("sha256"):
        out.append(f"{kind}: {rel(p, base)} is missing or not its manifest's sha256")
    rows = ent.get("step1_cache_rows")
    if not isinstance(rows, dict) or rows.get("equal") is not True:
        out.append(f"{kind}: the pools are not checked against step 1's cache rows ({rows})")
    if kind == "p1c2" and ent.get("construction") != CONSTRUCTION:
        out.append(f"{kind}: construction {ent.get('construction')}, not {CONSTRUCTION}")
    if carve == "fit":
        if ent.get("questions") != FIT_N or ent.get("carve_ids_sha256") != FIT_IDS_SHA:
            out.append(f"{kind}: {ent.get('questions')} questions hashing to {str(ent.get('carve_ids_sha256'))[:12]}, "
                       f"not W0's {FIT_N} ({FIT_IDS_SHA[:12]})")
    elif kind == "step4d":
        b = ent.get("step4b") or {}
        if b.get("sha256") != STEP4B_EVAL_SHA or not b.get("equal") or not all(b["equal"].values()):
            out.append(f"{kind}: not step 4b's filed arrays ({b})")
    else:
        c = ent.get("checks") or {}
        if not c or not all(c.values()):
            out.append(f"{kind}: P1e's checks {c}")
    return out


def s4d(check, threads, host):
    """W3 (or the check's first run): step 4d's union pools of webqsp's carve."""
    carve = "s1eval" if check else "fit"
    with on_wq_pools(check, host) as P:
        P.D4.pools_stage(["webqsp"], threads)
        d, man = Path(P.D4.POOLS), read_json(P.D4.MANIFEST)
    probs = pool_entry_problems(man["files"].get(f"webqsp__{carve}"), carve, d, "step4d")
    if probs:
        raise SystemExit("zwq s4d: " + "; ".join(probs))
    log(f"s4d webqsp/{carve}: filed at {rel(d)}, every check held")
    return 0


def pools(check, threads, host):
    """W4 (or the check's second run): P1c2's KIND_T pools of webqsp's carve, on W3's union pools."""
    carve = "s1eval" if check else "fit"
    with on_wq_pools(check, host) as P:
        P.PC2.pools_stage("webqsp", threads)
        d = Path(P.PC2.POOLS) / "webqsp"
    probs = pool_entry_problems(read_json(d / "pools.json")["files"].get(f"webqsp__{carve}"), carve, d, "p1c2")
    if probs:
        raise SystemExit("zwq pools: " + "; ".join(probs))
    log(f"pools webqsp/{carve}: filed at {rel(d)}, every check held")
    return 0


def kt(threads, host):
    """W8: P1c3's KT input of webqsp's fit carve, on W4's pools and W6's cache."""
    with on_wq_pools(False, host) as P:
        import p1c3_kt as PK
        if PK.PC2 is not P.PC2:
            raise SystemExit("zwq: p1c3_kt does not read p1c2_pools' roots")
        with patch(PK, KT=ROOT / ROOTS["kt"]):
            PK.feats_stage("webqsp", threads, 0, "fit")
    r = read_json(ROOT / ROOTS["kt"] / "webqsp" / "fit" / "record.json")
    if r.get("cache_root") != ROOTS["cache"] or r.get("questions") != FIT_N or r.get("declared_in") != KT_DECLARED:
        raise SystemExit(f"zwq kt: the record names {r.get('cache_root')}, {r.get('questions')} questions, "
                         f"{r.get('declared_in')}")
    log(f"kt webqsp/fit: {r['questions']} questions, {r['rows']} rows")
    return 0


def same_arrays(a, b):
    """Every key, dtype, shape and byte of two npz files."""
    import numpy as np
    with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
        ka, kb = set(za.files), set(zb.files)
        differ = []
        for k in sorted(ka & kb):
            x, y = za[k], zb[k]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                differ.append({"key": k, "dtype": [str(x.dtype), str(y.dtype)], "shape": [list(x.shape), list(y.shape)]})
    return {"keys": len(ka | kb), "only_new": sorted(ka - kb), "only_filed": sorted(kb - ka), "differ": differ,
            "equal": not (differ or ka ^ kb)}


def check(base=ROOT):
    """Section 4: the check's two runs (webqsp's s1eval carve through this file's patches) against step 4d's and P1c2's
    filed arrays. REPRODUCED needs every key, dtype, shape and byte; the manifest entries are reported beside."""
    new = {"step4d": base / CHECK_ROOTS["s4d"], "p1c2": base / CHECK_ROOTS["pools"] / "webqsp"}
    pairs = {}
    for k, (npz, man) in FILED.items():
        n_npz, n_man = new[k] / "webqsp__s1eval.npz", new[k] / "pools.json"
        for p in (n_npz, n_man, base / npz, base / man):
            if not p.is_file():
                raise SystemExit(f"zwq check: {rel(p, base)} missing")
        e_new = read_json(n_man)["files"]["webqsp__s1eval"]
        e_old = read_json(base / man)["files"]["webqsp__s1eval"]
        skip = ("file", "sha256", "seconds", "utc")
        pairs[k] = {"new": rel(n_npz, base), "filed": npz, "new_sha256": sha_file(n_npz),
                    "filed_sha256": sha_file(base / npz), "arrays": same_arrays(n_npz, base / npz),
                    "entry_differs_in": sorted(x for x in set(e_new) | set(e_old)
                                               if x not in skip and e_new.get(x) != e_old.get(x))}
    ok = all(v["arrays"]["equal"] for v in pairs.values())
    rec = {"stage": STAGE, "declared_in": DECLARED, "step": "check", "carve": "webqsp/s1eval",
           "status": "REPRODUCED" if ok else "NOT_REPRODUCED", "pairs": pairs,
           "rule": "every key, dtype, shape and byte of both filed arrays; file sha256s are reported, not compared",
           "zwq_sha256": sha_src(__file__), "utc": utc()}
    write_json(base / CHECK, rec)
    log(f"check: {rec['status']} ({', '.join(f'{k} {v['arrays']}' for k, v in pairs.items() if not v['arrays']['equal'])})")
    return 0 if ok else 2


# ── the look, the builds and the bases (W1, W5, W7) ──────────────────────────


def copy_bases(base=ROOT):
    """W1: step 1's two bases, each checked against its record, at W2's and W6's cache roots."""
    src = base / STEP1_CACHE
    files = {}
    for g in BASES:
        npz, js = src / f"basis_{g}.npz", src / f"basis_{g}.json"
        want, got = read_json(js)["npz_sha256"], sha_file(npz)
        if got != want:
            raise SystemExit(f"zwq: {rel(npz, base)}: sha256 {got[:12]}, its record {want[:12]}")
        files[npz.name], files[js.name] = got, sha_file(js)
    placed = {ROOTS[r]: {n: place(src / n, base / ROOTS[r] / n, sha, base) for n, sha in files.items()}
              for r in ("s1cache", "cache")}
    rec = {"stage": STAGE, "declared_in": DECLARED, "step": "W1", "source": STEP1_CACHE, "files": files,
           "placed": placed, "status": "OK", "zwq_sha256": sha_src(__file__), "utc": utc()}
    write_json(base / BASES_REC, rec)
    log(f"copy-bases: {placed}")
    return 0


def look(shard, host):
    """W5: look_p1c2's look of webqsp's fit carve on W4's pools, written under outputs/wq/p1c2/look."""
    import look_p1c2 as LP2         # first: look_x_six sets the thread variables before numpy and torch load
    if "fit" not in LP2.CARVES:
        raise SystemExit(f"zwq: look_p1c2 looks {LP2.CARVES}, not fit")
    argv = ["--dataset", "webqsp", "--carve", "fit"] + (["--host"] if host else []) + ["--full", "--shard", shard]
    with patch(LP2, POOLS=ROOT / ROOTS["pools"], LOOK=ROOT / ROOTS["plook"]):
        rc = LP2.main(argv)
    return rc or 0


def build(what, host):
    """W7: zp1c2's builds (rmatch's and zrc's chains, zlink's links) of webqsp's fit carve at WQ's roots."""
    import zp1c2 as Z2
    with patch(Z2, LOOK=ROOT / ROOTS["plook"], CACHE=ROOT / ROOTS["cache"], CH_RM=ROOT / ROOTS["rmatch"],
               CH_ZRC=ROOT / ROOTS["zrc"], LINKS=ROOT / ROOTS["links"]):
        return Z2.build(what, "webqsp", "fit", host)


# ── the overlay and the gate (W9, W10) ───────────────────────────────────────


def overlay(base=ROOT, carves=OVERLAID):
    """W9: every file of the other carves' folders (P1c2's caches, chains and links, P1c3's KT input) at WQ's roots, a
    hard link where the disk allows it, else a copy; each placed file is its source's bytes, each record byte-equal, and
    each folder's records list only files that hash to them."""
    import p1c2_cache_restore as RS             # stdlib only
    folders, problems = {}, []
    for ds, cv in carves:
        for kind in kinds_of((ds, cv)):
            key = f"{kind} {ds}/{cv}"
            src, dst = base / SRC[kind] / ds / cv, base / ROOTS[kind] / ds / cv
            if not src.is_dir():
                problems.append(f"{key}: {SRC[kind]}/{ds}/{cv} is missing")
                continue
            try:
                files = sorted(p for p in src.rglob("*") if p.is_file() and ".tmp" not in p.name)
                how, records = {}, {}
                for p in files:
                    r = p.relative_to(src)
                    sha = sha_file(p)
                    h = place(p, dst / r, sha, base)
                    how[h] = how.get(h, 0) + 1
                    if p.suffix == ".json" and p.name.startswith("record"):
                        if (dst / r).read_bytes() != p.read_bytes():
                            raise SystemExit(f"{key}: {r.as_posix()} is not its source's bytes")
                        records[r.as_posix()] = sha
                chk = RS.check_kind(kind, base / ROOTS[kind], only=f"{ds}/{cv}")
                if not chk["whole"]:
                    problems.append(f"{key}: not whole at {ROOTS[kind]} ({brief(chk)})")
                folders[key] = {"source": f"{SRC[kind]}/{ds}/{cv}", "at": f"{ROOTS[kind]}/{ds}/{cv}", "files": len(files),
                                "placed": how, "records": records, "whole": chk["whole"]}
            except SystemExit as e:
                problems.append(clean(e, base))
    rec = {"stage": STAGE, "declared_in": DECLARED, "step": "W9", "status": "FAIL" if problems else "OK",
           "carves": [f"{d}/{c}" for d, c in carves], "folders": folders, "problems": problems,
           "zwq_sha256": sha_src(__file__), "utc": utc()}
    write_json(base / OVERLAY, rec)
    log(f"overlay: {rec['status']}, {len(folders)} folders, {sum(f['files'] for f in folders.values())} files"
        + (f"; {problems}" if problems else ""))
    return 0 if not problems else 2


def look_problems(d, what, base=ROOT):
    """(why W0's or W5's look of webqsp's fit carve under d is not every question's, its records' sha256s): three shards
    of one --full look with no limit, whose chunks cover the carve, each chunk on disk, and ids.json the carve's."""
    recs = sorted(Path(d).glob("record*.json"))
    names = [p.name for p in recs]
    if names != [f"record_{i}of3.json" for i in range(3)]:
        return [f"{what}: look records {names}, not record_0of3 .. record_2of3"], {}
    out, chunks = [], []
    rr = [read_json(p) for p in recs]
    for p, r in zip(recs, rr):
        got = (r.get("dataset"), r.get("carve"), r.get("carve_ids_sha256"), r.get("carve_queries"), r.get("queries"),
               r.get("full"), r.get("limit"), r.get("n_chunks"))
        want = ("webqsp", "fit", FIT_IDS_SHA, FIT_N, FIT_N, True, None, rr[0].get("n_chunks"))
        if got != want:
            out.append(f"{what}: {p.name} holds {got}, not {want}")
        chunks += list(r.get("chunks") or [])
    n = int(rr[0].get("n_chunks") or 0)
    on_disk = {int(p.stem[1:]) for p in (Path(d) / "chunks").glob("c*.npz")}
    if sorted(chunks) != list(range(n)) or not set(range(n)) <= on_disk:
        out.append(f"{what}: the shards' chunks are not 0 .. {n - 1} once each, on disk")
    ids = Path(d) / "ids.json"
    if not ids.is_file() or len(read_json(ids)) != FIT_N:
        out.append(f"{what}: ids.json is missing or not {FIT_N} ids")
    return out, {p.name: sha_file(p) for p in recs}


def cache_problems(root, what, look_shas, RS, base=ROOT):
    """Why lean_cache's parts of webqsp's fit carve under root are not whole, every question once, on the look whose
    records hash to look_shas, or []."""
    chk = RS.check_kind("cache", root, only="webqsp/fit")
    out = [] if chk["whole"] else [f"{what}: the cache is not whole ({brief(chk)})"]
    q, ids = 0, []
    for p in sorted((Path(root) / "webqsp" / "fit").glob("part_*of*")):
        r = read_json(p / "record.json")
        q += int(r.get("queries") or 0)
        ids += list(r.get("ids") or [])
        lk = r.get("look") or {}
        if lk.get("carve_ids_sha256") != FIT_IDS_SHA or lk.get("carve_queries") != FIT_N:
            out.append(f"{what}: {p.name} was built on carve {str(lk.get('carve_ids_sha256'))[:12]} "
                       f"({lk.get('carve_queries')} questions)")
        if lk.get("records") != look_shas:
            out.append(f"{what}: {p.name} was built on other look records than {rel(root, base)}'s look")
    if q != FIT_N or len(ids) != FIT_N or len(set(ids)) != FIT_N:
        out.append(f"{what}: the parts hold {q} questions ({len(set(ids))} distinct ids), not {FIT_N}")
    return out


def part_dirs(root, ds, cv):
    return sorted(p for p in (Path(root) / ds / cv).glob("part_*of*") if p.is_dir())


def gate(base=ROOT):
    """W10: PASS when W1 to W9 and the check hold (section 3); FAIL names what does not. Exit 0 on PASS, 2 on FAIL."""
    import p1c2_cache_restore as RS
    problems, seen = [], {}

    def section(what, fn):
        try:
            problems.extend(fn() or [])
        except (Exception, SystemExit) as e:            # a missing or unreadable record is the stage's problem
            problems.append(f"{what}: {clean(repr(e) if not isinstance(e, SystemExit) else e, base)}")

    w = {}

    def w1():
        b = read_json(base / BASES_REC)
        seen["bases"] = sha_file(base / BASES_REC)
        out = [] if (b.get("status"), b.get("declared_in")) == ("OK", DECLARED) else ["W1: bases.json is not WQ's OK"]
        for g in BASES:
            w[g] = b["files"][f"basis_{g}.npz"]
            for r in ("s1cache", "cache"):
                for n in (f"basis_{g}.npz", f"basis_{g}.json"):
                    p = base / ROOTS[r] / n
                    if not p.is_file() or sha_file(p) != b["files"][n]:
                        out.append(f"W1: {ROOTS[r]}/{n} is missing or not step 1's")
        return out

    def w0_w2():
        out, w["w0"] = look_problems(base / ROOTS["look"] / "webqsp" / "fit", "W0", base)
        return out + cache_problems(base / ROOTS["s1cache"], "W2", w["w0"], RS, base)

    def w3():
        d = base / ROOTS["s4d"]
        seen["step4d_manifest"] = sha_file(d / "pools.json")
        return pool_entry_problems(read_json(d / "pools.json")["files"].get("webqsp__fit"), "fit", d, "W3 step4d", base)

    def w4():
        d = base / ROOTS["pools"] / "webqsp"
        seen["p1c2_manifest"] = sha_file(d / "pools.json")
        ent = read_json(d / "pools.json")["files"].get("webqsp__fit")
        w["w4"] = (ent or {}).get("sha256")
        out = pool_entry_problems(ent, "fit", d, "p1c2", base)
        return [f"W4 {x}" for x in out]

    def w5_w6():
        d = base / ROOTS["plook"] / "webqsp" / "fit"
        out, w["w5"] = look_problems(d, "W5", base)
        for i in range(3):
            p = d / f"pools_{i}of3.json"
            r = read_json(p) if p.is_file() else {}
            if r.get("pools_sha256") != w.get("w4") or r.get("construction") != CONSTRUCTION:
                out.append(f"W5: pools_{i}of3.json names {str(r.get('pools_sha256'))[:12]} ({r.get('construction')}), "
                           f"not W4's pools")
        return out + cache_problems(base / ROOTS["cache"], "W6", w["w5"], RS, base)

    def w7():
        return [f"W7: {k} of webqsp/fit is not whole ({brief(c)})" for k in ("rmatch", "zrc", "links")
                for c in [RS.check_kind(k, base / ROOTS[k], only="webqsp/fit")] if not c["whole"]]

    def w8():
        c = RS.check_kind("kt", base / ROOTS["kt"], only="webqsp/fit")
        out = [] if c["whole"] else [f"W8: the KT input is not whole ({brief(c)})"]
        rp = base / ROOTS["kt"] / "webqsp" / "fit" / "record.json"
        r = read_json(rp)
        seen["kt_record"] = sha_file(rp)
        parts = {p.name: sha_file(p / "record.json") for p in part_dirs(base / ROOTS["cache"], "webqsp", "fit")}
        got = (r.get("declared_in"), r.get("cache_root"), r.get("questions"), r.get("cache_parts") == parts)
        if got != (KT_DECLARED, ROOTS["cache"], FIT_N, True):
            out.append(f"W8: the KT record is {got[:3]}, cache parts W6's: {got[3]}")
        return out

    def w9():
        o = read_json(base / OVERLAY)
        seen["overlay"] = sha_file(base / OVERLAY)
        out = [] if (o.get("status"), o.get("declared_in")) == ("OK", DECLARED) else ["W9: overlay.json is not OK"]
        for carve in SCREEN:
            for k in kinds_of(carve):
                c = RS.check_kind(k, base / ROOTS[k], only="/".join(carve))
                if not c["whole"]:
                    out.append(f"W9: {k} {'/'.join(carve)} is not whole at {ROOTS[k]} ({brief(c)})")
        return out

    def bases_named():
        out = []
        for ds, cv in SCREEN:
            ps = part_dirs(base / ROOTS["cache"], ds, cv)
            if not ps:
                out.append(f"bases: no cache part of {ds}/{cv}")
            for p in ps:
                b = read_json(p / "record.json").get("bases") or {}
                if "2wiki" not in b or any(b[g].get("npz_sha256") != w.get(g) for g in b):
                    out.append(f"bases: {ds}/{cv}/{p.name} names {sorted(b)} not all W1's")
        return out

    def checked():
        c = read_json(base / CHECK)
        seen["check"] = sha_file(base / CHECK)
        return [] if c.get("status") == "REPRODUCED" else [f"check: {c.get('status')}"]

    for what, fn in (("W1", w1), ("W0/W2", w0_w2), ("W3", w3), ("W4", w4), ("W5/W6", w5_w6), ("W7", w7), ("W8", w8),
                     ("W9", w9), ("bases", bases_named), ("check", checked)):
        section(what, fn)
    rec = {"stage": STAGE, "declared_in": DECLARED, "step": "W10", "verdict": "FAIL" if problems else "PASS",
           "problems": problems, "records": seen, "look_records": {k: w.get(k) for k in ("w0", "w5")},
           "zwq_sha256": sha_src(__file__), "utc": utc()}
    write_json(base / GATE, rec)
    log(f"gate: {rec['verdict']}" + (f": {problems}" if problems else ""))
    return 0 if not problems else 2


def gate_passed(base=ROOT):
    p = base / GATE
    if not p.is_file():
        raise SystemExit(f"zwq: {GATE} missing (zwq.py gate)")
    rec = read_json(p)
    if rec.get("verdict") != "PASS" or rec.get("declared_in") != DECLARED:
        raise SystemExit(f"zwq: {GATE} is {rec.get('verdict')}, not WQ's PASS")
    return rec


# ── the fits: train, read, compare (section 5) ───────────────────────────────


def kt_loader(orig, base=ROOT):
    """zp1c3.load_kt for WQ's roots: a linked KT record (one of the ten other carves', written on P1c2's cache root)
    is handed that root, whose parts WQ's are byte for byte (its cache_parts are still checked); webqsp's fit carve's
    record names WQ's root and goes through as it is."""
    wq_cache, p1c2_cache = (base / ROOTS["cache"]).resolve(), base / SRC["cache"]

    def load_kt(c, root, cache_root, verify=True):
        if (c.ds, c.carve) in OVERLAID and Path(cache_root).resolve() == wq_cache:
            rp = Path(root) / c.ds / c.carve / "record.json"
            if rp.is_file() and read_json(rp).get("cache_root") == SRC["cache"]:
                cache_root = p1c2_cache
        return orig(c, root, cache_root, verify)

    return load_kt


@contextlib.contextmanager
def on_wq():
    """zp1c3p's and zp1c3's train and read on WQ's splits, carves and roots (section 5); restored after."""
    import zp1c3p as Z3P            # zrckp; through it zp1c3's zspk, zp1c2's roots, lean_gpu and lean_screen
    Z3 = Z3P.Z3
    Z2, LG, LC, S2, S = Z3.Z2, Z3.LG, Z3.LC, Z3.S2, Z3.S
    if tuple(LG.EVAL_ORDER) != EVAL_ORDER or S.FLOOR != FLOOR:
        raise SystemExit(f"zwq: lean_gpu reads {LG.EVAL_ORDER} at floor {S.FLOOR}, not WQ's {EVAL_ORDER} at {FLOOR}")
    if any(LG.FIT_CARVE.get(d) != cv for d, cv in FIT_CARVE.items() if d != "webqsp"):
        raise SystemExit(f"zwq: lean_gpu's fit carves are {LG.FIT_CARVE}")
    kt_root = ROOT / ROOTS["kt"]
    orig_records = Z3.kt_records
    ns = types.SimpleNamespace(Z3P=Z3P, Z3=Z3, Z2=Z2, LG=LG, LC=LC, S2=S2, S=S, SR=Z3.SR)
    with contextlib.ExitStack() as st:
        st.enter_context(patch(LG, FIT_CARVE={**LG.FIT_CARVE, "webqsp": "fit"}, FITS={**LG.FITS, **SPLITS}))
        st.enter_context(patch(Z3, SPLITS=tuple(SPLITS), load_kt=kt_loader(Z3.load_kt),
                               kt_records=lambda carves, root=kt_root: orig_records(carves, root)))
        st.enter_context(patch(Z3P, SPLITS=tuple(SPLITS)))
        st.enter_context(patch(Z2, LOOK=ROOT / ROOTS["plook"], CACHE=ROOT / ROOTS["cache"], CH_RM=ROOT / ROOTS["rmatch"],
                               CH_ZRC=ROOT / ROOTS["zrc"], LINKS=ROOT / ROOTS["links"]))
        st.enter_context(patch(Z3.KTChainCarveZRC, KT_ROOT=kt_root))
        st.enter_context(patch(Z3.KTLinkCarveBase, KT_ROOT=kt_root))
        yield ns


def fit_dir(name, base=ROOT):
    return base / "outputs" / "screen" / "fits" / name


def wq_why(name, base=ROOT):
    """Why the fit record of name is not WQ's fit of its arm, seed and split, or None."""
    arm, seed, split = FITS[name]
    sj = fit_dir(name, base) / "screen.json"
    if not sj.is_file():
        return f"{name} has no record (outputs/screen/fits/{name}/screen.json)"
    s = read_json(sj)
    w = s.get("wq") or {}
    got = (s.get("arm"), s.get("seed"), s.get("split"), w.get("declared_in"), w.get("arm"), w.get("seed"), w.get("split"))
    if got != (arm, seed, split, DECLARED, arm, seed, split) or STAMP[arm] not in s:
        return f"{name}'s record is {got}, {STAMP[arm]} stamp {STAMP[arm] in s}; not WQ's {arm} seed {seed} on {split}"
    return None


def cache_root_ok(cache_root):
    if parts_of(cache_root) != parts_of(ROOTS["cache"]):
        raise SystemExit(f"zwq: train and read take --cache-root {ROOTS['cache']}, not {cache_root}")


def train(argv):
    """A fit of the base set: zp1c3p's train (zrckp, under WQ's fit name) or zp1c3's (zspk), on WQ's split, then the
    record's wq block."""
    ap = argparse.ArgumentParser(prog="zwq.py train")
    ap.add_argument("--split", required=True, choices=tuple(SPLITS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--cache-root", required=True)
    a = ap.parse_args(argv)
    if a.name != fit_name(a.arm, a.seed, a.split):
        raise SystemExit(f"zwq: {a.arm}'s seed {a.seed} fit of {a.split} is {fit_name(a.arm, a.seed, a.split)}, "
                         f"not {a.name}")
    cache_root_ok(a.cache_root)
    gate_passed()
    args = ["train", "--name", a.name, "--arm", a.arm, "--seed", str(a.seed), "--device", a.device,
            "--cache-root", ROOTS["cache"]] + (["--host"] if a.host else [])
    with on_wq() as W:
        if a.arm == "zrckp":
            with patch(W.Z3P, FIT={**W.Z3P.FIT, a.split: a.name}):
                rc = W.Z3P.train(args, a.split)
        else:
            rc = W.Z3.train(args, a.split)
        if rc == 0:
            sj = fit_dir(a.name) / "screen.json"
            rec = read_json(sj)
            rec["zwq_sha256"] = sha_src(__file__)
            rec["wq"] = {"declared_in": DECLARED, "split": a.split, "seed": a.seed, "arm": a.arm,
                         "trains_on": list(SPLITS[a.split]), "zero_shot": list(ZERO_SHOT[a.split]), "roots": dict(ROOTS),
                         "gate": sha_file(ROOT / GATE), "overlay": sha_file(ROOT / OVERLAY)}
            W.LC.write_json(sj, rec)
            log(f"train {a.name}: done, stamped")
    return rc


def read(argv):
    """A fit's read of the six s1eval carves (zp1c3p's or zp1c3's read on WQ's roots)."""
    ap = argparse.ArgumentParser(prog="zwq.py read")
    ap.add_argument("--name", required=True, choices=sorted(FITS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--cache-root", required=True)
    a = ap.parse_args(argv)
    cache_root_ok(a.cache_root)
    gate_passed()
    why = wq_why(a.name)
    if why:
        raise SystemExit(f"zwq: {why}")
    args = ["--name", a.name, "--device", a.device, "--cache-root", ROOTS["cache"]] + (["--host"] if a.host else [])
    with on_wq() as W:
        return (W.Z3P.read if FITS[a.name][0] == "zrckp" else W.Z3.read)(args)


def run_compare(new, base_fit, out):
    if not os.path.samefile(Path.cwd(), ROOT):
        raise SystemExit("zwq: compare runs from the repository's root")
    with on_wq() as W:
        return W.S2.main(["compare", "--new", new, "--base", base_fit, "--out", out])


def compare(argv):
    """A null comparison (section 6): WQ's seed-1 or seed-2 fit against its seed-0 fit of the same arm and split alone."""
    ap = argparse.ArgumentParser(prog="zwq.py compare")
    ap.add_argument("--new", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    n = parts_of(a.new)
    if len(n) != 4 or n[:3] != ("outputs", "screen", "fits") or n[3] not in FITS or FITS[n[3]][1] not in NULL_SEEDS:
        raise SystemExit(f"zwq: compare --new takes outputs/screen/fits/ and a seed-1 or seed-2 WQ fit, not {a.new}")
    arm, _seed, split = FITS[n[3]]
    b0 = fit_name(arm, 0, split)
    if parts_of(a.base) != ("outputs", "screen", "fits", b0):
        raise SystemExit(f"zwq: {n[3]} is compared with outputs/screen/fits/{b0} alone, not {a.base}")
    if parts_of(a.out) != ("outputs", "screen", n[3]):
        raise SystemExit(f"zwq: {n[3]}'s comparison is outputs/screen/{n[3]}, not {a.out}")
    for name in (n[3], b0):
        why = wq_why(name)
        if why:
            raise SystemExit(f"zwq: {why}")
    return run_compare(a.new, a.base, a.out)


def beside(arm, split):
    """Section 8: WQ's seed-0 fit against the lean pair's card fit of the split's wiki side, beside; it decides nothing."""
    name = fit_name(arm, 0, split)
    why = wq_why(name)
    if why:
        raise SystemExit(f"zwq: {why}")
    old = f"{OLD_CARD[arm]}{SUF[split]}"
    if not (fit_dir(old) / "read.json").is_file():
        raise SystemExit(f"zwq: outputs/screen/fits/{old} has no read")
    return run_compare(f"outputs/screen/fits/{name}", f"outputs/screen/fits/{old}", f"{BESIDE}/{name}")


# ── the floors and the summary (sections 6 and 7) ────────────────────────────


def null_compares(arm, base=ROOT):
    """{(split, seed): {dataset: row}} from the arm's four null comparisons, or SystemExit naming what is wrong."""
    got = {}
    for split in SPLITS:
        b0 = fit_name(arm, 0, split)
        for seed in NULL_SEEDS:
            name = fit_name(arm, seed, split)
            p = base / "outputs" / "screen" / f"{name}.json"
            if not p.is_file():
                raise SystemExit(f"outputs/screen/{name}.json missing")
            rec = read_json(p)
            bases = rec.get("bases") or []
            if len(bases) != 1 or parts_of(bases[0])[-3:] != ("screen", "fits", b0):
                raise SystemExit(f"outputs/screen/{name}.json is decided against {bases}, not {b0} alone")
            if parts_of(rec.get("new", ""))[-3:] != ("screen", "fits", name):
                raise SystemExit(f"outputs/screen/{name}.json reads {rec.get('new')}, not {name}")
            if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
                raise SystemExit(f"outputs/screen/{name}.json reads {rec.get('candidate')} on {rec.get('carve')}")
            if sorted(rec.get("trained_on") or []) != sorted(SPLITS[split]):
                raise SystemExit(f"outputs/screen/{name}.json trained on {rec.get('trained_on')}, not {split}'s "
                                 f"{sorted(SPLITS[split])}")
            for f in (name, b0):
                why = wq_why(f, base)
                if why:
                    raise SystemExit(why)
            rows = rec["by_base"][bases[0]]["rows"]
            if sorted(r["dataset"] for r in rows) != sorted(EVAL_ORDER):
                raise SystemExit(f"outputs/screen/{name}.json reads {[r['dataset'] for r in rows]}, not the six")
            got[(split, seed)] = {r["dataset"]: r for r in rows}
    return got


def floors(arm, base=ROOT, floor_of=None):
    """Section 6: outputs/wq/floors-<arm>.json and .md, each read's floor from its two seed-only differences
    (screen_recall.floor_of). Exit 0 with status OK; 2 after writing a REFUSED record."""
    if floor_of is None:
        import screen_recall as SR
        floor_of = SR.floor_of
    head = {"stage": STAGE, "declared_in": DECLARED, "arm": arm,
            "fits": {sp: [fit_name(arm, sd, sp) for sd in SEEDS] for sp in SPLITS}}
    try:
        got = null_compares(arm, base)
        reads = []
        for split in SPLITS:
            for ds in EVAL_ORDER:
                r1, r2 = got[(split, 1)][ds], got[(split, 2)][ds]
                if r1["base"][0] != r2["base"][0]:
                    raise SystemExit(f"{split} {ds}: the two seeds are compared with different seed-0 R@5 "
                                     f"({r1['base'][0]}, {r2['base'][0]})")
                d1, d2 = r1["delta"][0], r2["delta"][0]
                f = floor_of(d1, d2)
                reads.append({"split": split, "dataset": ds, "read": r1.get("read"), "seed0": r1["base"][0],
                              "d1": round(d1, 6), "d2": round(d2, 6),
                              "spread": round(math.sqrt((d1 * d1 + d2 * d2) / 2.0), 6), "floor": round(f, 6),
                              "raised": f > FLOOR})
        raised = [f"{r['split']} {r['dataset']}" for r in reads if r["raised"]]
        rec = {**head, "status": "OK", "floor_min": FLOOR, "reads": reads, "raised": raised,
               "rule": "floor(r) = max(0.0075, 2 * sqrt((D_1(r)^2 + D_2(r)^2) / 2)), D_s(r) = seed s's R@5 minus seed "
                       "0's on read r (screen_recall.floor_of)", "zwq_sha256": sha_src(__file__), "utc": utc()}
        md = [f"# WQ's floors on {arm}", "",
              f"{arm} fitted on each WQ split with seeds 0, 1 and 2; seeds 1 and 2 each compared with seed 0 on the six "
              "s1eval carves (p@swa, R@5). floor = max(0.0075, 2 sqrt((D1^2 + D2^2) / 2)). Bold: above 0.0075.", "",
              "| split | dataset | read | seed 0 R@5 | D1 | D2 | spread | floor |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in reads:
            fl = f"{r['floor']:.4f}"
            md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['seed0']:.4f} | {r['d1']:+.4f} | "
                      f"{r['d2']:+.4f} | {r['spread']:.4f} | {'**' + fl + '**' if r['raised'] else fl} |")
        md += ["", f"Floors above 0.0075: {len(raised)} of {len(reads)}" + (f" ({', '.join(raised)})." if raised else ".")]
        rc = 0
    except (SystemExit, KeyError, IndexError, TypeError, ValueError) as e:
        rec = {**head, "status": "REFUSED", "reason": clean(e, base), "zwq_sha256": sha_src(__file__), "utc": utc()}
        md = [f"# WQ's floors on {arm}: REFUSED", "", clean(e, base)]
        rc = 2
    write(base / FLOORS[arm], rec, md)
    log(f"floors {arm}: {rec['status']}" + (f" ({rec.get('reason')})" if rc else f", {len(rec['raised'])} raised"))
    return rc


def summary(base=ROOT):
    """Section 7: BASE_READY when the gate passed on a reproduced check, the twelve fits are trained, stamped and read,
    the eight null comparisons are filed and both floors are OK; NOT_READY names what is missing."""
    missing = []
    for p, key, want in ((GATE, "verdict", "PASS"), (CHECK, "status", "REPRODUCED")):
        r = read_json(base / p) if (base / p).is_file() else {}
        if r.get(key) != want:
            missing.append(f"{p}: {r.get(key)}, not {want}")
    six = {f"{d}=s1eval" for d in EVAL_ORDER}
    for name in FITS:
        why = wq_why(name, base)
        if why:
            missing.append(why)
            continue
        if read_json(fit_dir(name, base) / "screen.json").get("rc") != 0:
            missing.append(f"{name}: its train did not exit 0")
        rj = fit_dir(name, base) / "read.json"
        if not rj.is_file() or set(read_json(rj).get("carves") or {}) != six:
            missing.append(f"{name}: not read on the six s1eval carves")
    for arm in ARMS:
        try:
            null_compares(arm, base)
        except (SystemExit, KeyError, TypeError) as e:
            missing.append(f"{arm}'s null comparisons: {clean(e, base)}")
        p = base / (FLOORS[arm] + ".json")
        st = read_json(p).get("status") if p.is_file() else None
        if st != "OK":
            missing.append(f"{FLOORS[arm]}.json: {st}")
    besides = {}
    for arm in ARMS:
        for split in SPLITS:
            p = base / BESIDE / f"{fit_name(arm, 0, split)}.json"
            if p.is_file():
                besides[fit_name(arm, 0, split)] = read_json(p).get("verdict")
    status = "NOT_READY" if missing else "BASE_READY"
    rec = {"stage": STAGE, "declared_in": DECLARED, "status": status, "missing": missing, "fits": sorted(FITS),
           "beside": besides, "zwq_sha256": sha_src(__file__), "utc": utc()}
    md = [f"# WQ's base set: {status}", ""]
    if missing:
        md += ["Missing:", ""] + [f"- {m}" for m in missing]
    else:
        md += ["| arm | split | dataset | read | seed 0 R@5 | floor |", "|---|---|---|---|---:|---:|"]
        for arm in ARMS:
            for r in read_json(base / (FLOORS[arm] + ".json"))["reads"]:
                md.append(f"| {arm} | {r['split']} | {r['dataset']} | {r['read']} | {r['seed0']:.4f} | {r['floor']:.4f} |")
    if besides:
        md += ["", "Beside (section 8, deciding nothing): " + "; ".join(f"{k} {v}" for k, v in besides.items()) + "."]
    write(base / SUMMARY, rec, md)
    log(f"summary: {status}" + (f": {missing}" if missing else ""))
    return 0 if not missing else 2


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile

    import numpy as np
    import p1c2_cache_restore as RS

    # patch: set for the block, restored after a failure, a missing name refused
    ns = types.SimpleNamespace(a=1, b=2)
    try:
        with patch(ns, a=10, b=20):
            assert (ns.a, ns.b) == (10, 20)
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert (ns.a, ns.b) == (1, 2)
    assert "has no" in must_stop(lambda: patch(ns, c=3).__enter__())

    # names, carves and kinds
    assert fit_name("zrckp", 0, "WQ-musique") == "scr-wq-zrckp-s0" and fit_name("zspk", 2, "WQ-hotpotqa") == "scr-wq-zspk-s2-hp"
    assert FITS["scr-wq-zspk-s1-hp"] == ("zspk", 1, "WQ-hotpotqa") and NEW not in OVERLAID
    assert kinds_of(("metaqa", "s1eval")) == ("cache", "links", "kt", "rmatch", "zrc") and kinds_of(("squad", "fit")) == (
        "cache", "links", "kt")
    assert "not scr-wq-zrckp-s1" in must_stop(train, ["--split", "WQ-musique", "--name", "scr-wq-zrckp-s1", "--arm",
                                                      "zrckp", "--seed", "0", "--cache-root", ROOTS["cache"]])
    assert "--cache-root outputs/wq/p1c2/cache" in must_stop(train, ["--split", "WQ-hotpotqa", "--name",
                                                                     "scr-wq-zspk-s0-hp", "--arm", "zspk", "--seed", "0",
                                                                     "--cache-root", "outputs/p1c2/cache"])
    fake = "\\".join(["C:", "Users", "someone", "x", "y.json missing"])      # a home folder, joined here
    assert clean(fake, Path("D:/elsewhere")) == "<home>\\x\\y.json missing"

    # the Opened wrapper: webqsp's fit carve added; a differing one refused; the rest untouched
    class Fake:
        def __init__(self):
            self.carves = {"per_dataset": {"webqsp": {"eval": {"name": "s1eval"}}, "squad": {"fit": {"name": "fit"}}}}

    o = with_fit(Fake)()
    assert o.carves["per_dataset"]["webqsp"]["fit"] == {"name": "fit", "n": FIT_N, "sha256": FIT_IDS_SHA}
    assert o.carves["per_dataset"]["squad"] == {"fit": {"name": "fit"}} and type(o).__name__ == "WQFake"

    class Fake2(Fake):
        def __init__(self):
            super().__init__()
            self.carves["per_dataset"]["webqsp"]["fit"] = {"name": "fit", "n": 7, "sha256": "x"}

    assert "not W0's" in must_stop(with_fit(Fake2))

    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        # place: a link (or copy), then present; a differing file at dst or a wrong source sha256 stops
        s = T / "src.bin"
        s.write_bytes(b"abc")
        sh = sha_file(s)
        assert place(s, T / "d" / "x.bin", sh, T) in ("linked", "copied") and (T / "d" / "x.bin").read_bytes() == b"abc"
        assert place(s, T / "d" / "x.bin", sh, T) == "present"
        (T / "d" / "y.bin").write_bytes(b"abd")
        assert "is present and is not" in must_stop(place, s, T / "d" / "y.bin", sh, T)
        assert "is not the sha256" in must_stop(place, s, T / "d" / "z.bin", "0" * 64, T)

        # same_arrays: equal, a byte, a dtype, a key
        a1, a2 = T / "a1.npz", T / "a2.npz"
        np.savez(a1, x=np.arange(4, dtype=np.int32), y=np.array(["p", "q"]))
        np.savez(a2, x=np.arange(4, dtype=np.int32), y=np.array(["p", "q"]))
        assert same_arrays(a1, a2)["equal"]
        np.savez(a2, x=np.array([0, 1, 2, 4], dtype=np.int32), y=np.array(["p", "q"]))
        assert [d["key"] for d in same_arrays(a1, a2)["differ"]] == ["x"]
        np.savez(a2, x=np.arange(4, dtype=np.int64), y=np.array(["p", "q"]))
        assert not same_arrays(a1, a2)["equal"]
        np.savez(a2, x=np.arange(4, dtype=np.int32), y=np.array(["p", "q"]), z=np.zeros(1))
        r = same_arrays(a1, a2)
        assert r["only_filed"] == ["z"] and not r["equal"]

        # the overlay on toy roots: three carves (one chained), every file placed, records equal, folders whole
        B = T / "repo"

        def toy_unit(kind, ds, cv, payload):
            if kind == "kt":
                u = B / SRC[kind] / ds / cv
                u.mkdir(parents=True)
                (u / "kt.npy").write_bytes(payload)
                (u / "n.npy").write_bytes(payload + b"n")
                (u / "ids.json").write_text('["q1"]', encoding="utf-8")
                rec_ = {"arrays": {"kt": {"sha256": sha_file(u / "kt.npy")}, "n": {"sha256": sha_file(u / "n.npy")}},
                        "ids_sha256": sha_file(u / "ids.json"), "cache_root": SRC["cache"]}
            else:
                u = B / SRC[kind] / ds / cv / "part_0of1"
                u.mkdir(parents=True)
                (u / "a.npy").write_bytes(payload)
                rec_ = {"arrays": {"a": {"sha256": sha_file(u / "a.npy")}}}
            write_json(u / "record.json", rec_)

        toys = [("squad", "fit"), ("metaqa", "s1eval"), ("webqsp", "s1eval")]
        for i, c in enumerate(toys):
            for k in kinds_of(c):
                toy_unit(k, c[0], c[1], f"{k}{i}".encode())
        assert overlay(B, toys) == 0
        ov = read_json(B / OVERLAY)
        assert ov["status"] == "OK" and len(ov["folders"]) == 3 + 5 + 5
        assert ov["folders"]["kt metaqa/s1eval"]["records"] == {"record.json": sha_file(B / SRC["kt"] / "metaqa/s1eval/record.json")}
        assert (B / ROOTS["zrc"] / "webqsp/s1eval/part_0of1/a.npy").read_bytes() == b"zrc2"
        assert overlay(B, toys) == 0 and read_json(B / OVERLAY)["folders"]["cache squad/fit"]["placed"] == {"present": 2}
        (B / ROOTS["links"] / "squad/fit/part_0of1/a.npy").unlink()
        (B / ROOTS["links"] / "squad/fit/part_0of1/a.npy").write_bytes(b"other")
        assert overlay(B, toys + [("2wiki", "fit")]) == 2
        pr = read_json(B / OVERLAY)["problems"]
        assert len(pr) == 4 and any("is present and is not" in x for x in pr) and any("2wiki/fit is missing" in x for x in pr)
        assert str(B) not in json.dumps(read_json(B / OVERLAY)) and B.as_posix() not in json.dumps(read_json(B / OVERLAY))

        # the KT loader: a linked carve's record (P1c2's cache root) is handed that root; the new carve's is not
        calls = []

        def orig(c, root, cache_root, verify=True):
            calls.append(Path(cache_root))
            return "ok"

        ld = kt_loader(orig, B)
        c_old = types.SimpleNamespace(ds="metaqa", carve="s1eval")
        assert ld(c_old, B / ROOTS["kt"], B / ROOTS["cache"]) == "ok" and calls[-1] == B / SRC["cache"]
        ld(c_old, B / ROOTS["kt"], B / "outputs/elsewhere")
        assert calls[-1] == B / "outputs/elsewhere"
        nk = B / ROOTS["kt"] / "webqsp" / "fit"
        nk.mkdir(parents=True)
        write_json(nk / "record.json", {"cache_root": ROOTS["cache"]})
        ld(types.SimpleNamespace(ds="webqsp", carve="fit"), B / ROOTS["kt"], B / ROOTS["cache"])
        assert calls[-1] == B / ROOTS["cache"]

        # the looks: three shards covering the chunks, ids.json, the records' sha256s
        L = T / "look"
        (L / "chunks").mkdir(parents=True)
        for j in range(5):
            (L / "chunks" / f"c{j}.npz").write_bytes(b"c")
        write_json(L / "ids.json", [f"q{j}" for j in range(FIT_N)])
        for i in range(3):
            write_json(L / f"record_{i}of3.json", {"dataset": "webqsp", "carve": "fit", "carve_ids_sha256": FIT_IDS_SHA,
                                                   "carve_queries": FIT_N, "queries": FIT_N, "full": True, "limit": None,
                                                   "n_chunks": 5, "chunks": list(range(i, 5, 3))})
        pr, shas = look_problems(L, "W0", T)
        assert pr == [] and sorted(shas) == [f"record_{i}of3.json" for i in range(3)]
        (L / "chunks" / "c4.npz").unlink()
        assert "on disk" in look_problems(L, "W0", T)[0][0]

        # the cache parts: whole, every question once, on the look's records
        C = T / "cache"
        for i, (lo, hi) in enumerate(((0, 600), (600, FIT_N))):
            u = C / "webqsp" / "fit" / f"part_{i}of2"
            u.mkdir(parents=True)
            (u / "x.npy").write_bytes(b"x%d" % i)
            write_json(u / "record.json", {"queries": hi - lo, "ids": [f"q{j}" for j in range(lo, hi)],
                                           "look": {"records": shas, "carve_ids_sha256": FIT_IDS_SHA, "carve_queries": FIT_N},
                                           "arrays": {"x": {"sha256": sha_file(u / "x.npy")}}})
        assert cache_problems(C, "W2", shas, RS, T) == []
        assert any("other look records" in x for x in cache_problems(C, "W2", {"record_0of3.json": "z"}, RS, T))

        # the floors from toy comparisons; a comparison trained on other sets is refused
        def toy_fit(name):
            arm, seed, split = FITS[name]
            write_json(fit_dir(name, B) / "screen.json",
                       {"arm": arm, "seed": seed, "split": split, "rc": 0, STAMP[arm]: {},
                        "wq": {"declared_in": DECLARED, "arm": arm, "seed": seed, "split": split}})

        def toy_compare(arm, split, seed, d, trained=None):
            name, b0 = fit_name(arm, seed, split), fit_name(arm, 0, split)
            rows = [{"dataset": ds, "read": "in-domain" if ds in SPLITS[split] else "zero-shot",
                     "base": [0.6, 0, 0], "delta": [d if ds == "webqsp" else 0.001 * seed, 0, 0]} for ds in EVAL_ORDER]
            b = f"outputs\\screen\\fits\\{b0}"
            write_json(B / "outputs" / "screen" / f"{name}.json",
                       {"new": f"outputs\\screen\\fits\\{name}", "bases": [b], "candidate": "p@swa", "carve": "s1eval",
                        "trained_on": sorted(trained or SPLITS[split]), "by_base": {b: {"rows": rows}}})

        fo = lambda d1, d2: max(FLOOR, 2.0 * math.sqrt((d1 * d1 + d2 * d2) / 2.0))  # noqa: E731
        for name in FITS:
            toy_fit(name)
        for split in SPLITS:
            for seed, d in ((1, 0.01), (2, -0.02)):
                toy_compare("zrckp", split, seed, d)
        assert floors("zrckp", B, fo) == 0
        fr = read_json(B / (FLOORS["zrckp"] + ".json"))
        w_ = {(r["split"], r["dataset"]): r for r in fr["reads"]}
        assert abs(w_[("WQ-musique", "webqsp")]["floor"] - 2 * math.sqrt((0.01 ** 2 + 0.02 ** 2) / 2)) < 1e-6
        assert w_[("WQ-hotpotqa", "squad")]["floor"] == FLOOR and len(fr["raised"]) == 2
        toy_compare("zrckp", "WQ-hotpotqa", 2, -0.02, trained=("metaqa", "squad", "musique", "2wiki"))
        assert floors("zrckp", B, fo) == 2 and "trained on" in read_json(B / (FLOORS["zrckp"] + ".json"))["reason"]
        assert floors("zspk", B, fo) == 2 and "missing" in read_json(B / (FLOORS["zspk"] + ".json"))["reason"]

        # the summary names what is missing
        assert summary(B) == 2 and any("gate.json" in m for m in read_json(B / (SUMMARY + ".json"))["missing"])
    print("selftest: patch restores; names, carves and kinds; the Opened wrapper; place links, keeps and refuses; "
          "same_arrays; the overlay places, keeps and refuses; the KT loader's root; the looks' and caches' checks; the "
          "floors and their refusals; the summary's missing list")

    # the patch points exist in the committed modules (imports torch and numba)
    with on_wq() as W:
        assert W.LG.FITS["WQ-musique"] == SPLITS["WQ-musique"] and W.LG.FIT_CARVE["webqsp"] == "fit"
        assert W.Z3.SPLITS == tuple(SPLITS) and W.Z3P.SPLITS == tuple(SPLITS)
        assert W.Z2.CACHE == ROOT / ROOTS["cache"] and W.Z3.KTChainCarveZRC.KT_ROOT == ROOT / ROOTS["kt"]
        assert W.Z3.train_carves("WQ-hotpotqa") == [(d, FIT_CARVE[d]) for d in SPLITS["WQ-hotpotqa"]]
        assert W.LG.basis_of(W.LG.FITS["WQ-musique"]) == "2wiki" == W.LG.basis_of(W.LG.FITS["WQ-hotpotqa"])
        Z3, Z2, LG = W.Z3, W.Z2, W.LG
    assert "WQ-musique" not in LG.FITS and "webqsp" not in LG.FIT_CARVE and Z3.SPLITS == ("L-musique", "L-hotpotqa")
    assert Z2.CACHE == ROOT / "outputs" / "p1c2" / "cache" and Z3.KTLinkCarveBase.KT_ROOT == Z3.KT_DIR
    with on_wq_pools() as P:
        assert P.C4.carves_of("webqsp") == ("fit",) and P.C4.carves_of("squad") == ("fit", "select", "s1sel", "s1eval")
        assert P.PC2.carves_for("webqsp") == ["fit"] and P.C4.STEP1_CACHE == ROOT / ROOTS["s1cache"]
        assert P.D4.MANIFEST == ROOT / ROOTS["s4d"] / "pools.json" and P.S4.Opened.__name__ == "WQOpened"
        import p1c3_kt as PK
        assert PK.PC2 is P.PC2 and hasattr(PK, "KT")
        C4, PC2, S4 = P.C4, P.PC2, P.S4
    assert C4.carves_of("webqsp") == ("s1eval",) and PC2.carves_for("webqsp") == ["s1eval"] and S4.Opened.__name__ == "Opened"
    with on_wq_pools(check=True) as P:
        assert P.C4.carves_of("webqsp") == ("s1eval",) and P.PC2.POOLS == ROOT / CHECK_ROOTS["pools"]
        assert P.D4.POOLS == ROOT / CHECK_ROOTS["s4d"] and P.C4.STEP1_CACHE == ROOT / STEP1_CACHE
    import zp1c2
    with patch(zp1c2, LOOK=1, CACHE=2, CH_RM=3, CH_ZRC=4, LINKS=5):
        pass
    print("selftest: on_wq and on_wq_pools patch lean_gpu, zp1c3, zp1c3p, zp1c2, step 4c, step 4d and P1c2's pools as "
          "declared, and restore them")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = (argv[0], argv[1:]) if argv else (None, [])
    if cmd == "--selftest":
        return selftest()
    if cmd == "train":
        return train(rest)
    if cmd == "read":
        return read(rest)
    if cmd == "compare":
        return compare(rest)
    ap = argparse.ArgumentParser(prog=f"zwq.py {cmd}")
    if cmd in ("s4d", "pools", "kt"):
        if cmd != "kt":
            ap.add_argument("--check", action="store_true")
        ap.add_argument("--threads", type=int, required=True)
        ap.add_argument("--host", action="store_true")
        a = ap.parse_args(rest)
        if cmd == "kt":
            return kt(a.threads, a.host)
        return (s4d if cmd == "s4d" else pools)(a.check, a.threads, a.host)
    if cmd == "look":
        ap.add_argument("--shard", required=True, choices=("0/3", "1/3", "2/3"))
        ap.add_argument("--host", action="store_true")
        a = ap.parse_args(rest)
        return look(a.shard, a.host)
    if cmd == "build":
        ap.add_argument("--what", required=True, choices=("rmatch", "zrc", "links"))
        ap.add_argument("--host", action="store_true")
        a = ap.parse_args(rest)
        return build(a.what, a.host)
    if cmd == "floors":
        ap.add_argument("--arm", required=True, choices=ARMS)
        return floors(ap.parse_args(rest).arm)
    if cmd == "beside":
        ap.add_argument("--arm", required=True, choices=ARMS)
        ap.add_argument("--split", required=True, choices=tuple(SPLITS))
        a = ap.parse_args(rest)
        return beside(a.arm, a.split)
    simple = {"copy-bases": copy_bases, "check": check, "overlay": overlay, "gate": gate, "summary": summary}
    if cmd in simple:
        ap.parse_args(rest)
        return simple[cmd]()
    raise SystemExit("zwq: copy-bases, s4d, pools, check, look, build, kt, overlay, gate, train, read, compare, floors, "
                     "beside, summary, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
