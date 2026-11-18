"""ENC E2 (docs/ENC_ENCODER_COMPARISON.md section 12.9): the small worlds, one per arm (frozen, mpnet, qwen3e06).

A world is a sibling tree of the workspace built by sworld.py's make (SWP's S-world code, not edited): copies of the
record's code, junctions to the record's other folders, and folders of its own. Every command in a world runs with the
start-up hook e2hook/sitecustomize.py (E2HOOK names the spec), which:
- logs every file the process opens (the audit);
- refuses reads of the record's files that the arm replaces;
- refuses writes outside the world;
- swaps, in the package loader, the dense vectors, the dense lists and the knn family of the arm's datasets for the
  arm's own.
The frozen world has the audit and the write guard, and no swap.

    python outputs/mp_unified/zenc2w.py pad --enc mpnet --ds squad          (the arm's vectors, zero-padded to 1536, in the
                                                                             package's layout, and its dense lists)
    python outputs/mp_unified/zenc2w.py world --enc mpnet                   (make or refresh the world)
    python outputs/mp_unified/zenc2w.py run --enc mpnet --tag T -- CMD      (verify, then CMD in the world with the hook)
    python outputs/mp_unified/zenc2w.py audit --enc mpnet --tag T           (what a run read, by kind)
    python outputs/mp_unified/zenc2w.py selftest

Outputs only under outputs/enc/e2/ and in the worlds.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = HERE.parents[1]
if str(ROOT / "outputs" / "host_ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "outputs" / "host_ops"))
import sworld as S  # noqa: E402  (the S-world's code, not edited)
import zenc2 as Z2  # noqa: E402

DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 12.9"
OUT = ROOT / "outputs" / "enc" / "e2"
ARMS = ("frozen", "mpnet", "qwen3e06")
SMALL = ("squad", "musique")                                  # the small worlds' datasets (section 12.9)
DIM = 1536
HOOK = HERE / "e2hook"
OWN = tuple(S.OWN) + ("pko", "d83", "abl")                    # the world's own folders (abl: filled by deep_own)
DEEP = {"abl": ("D83F", "D83E")}                              # own children of an own folder; its other children are junctions
COPY_CAP = 50 * 2 ** 20                                       # a plain file of a deep-own folder is copied up to this size


def log(msg):
    print(f"[zenc2w {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc2w: {msg}")


def world_dir(enc, root=ROOT):
    return Path(root).parent / f"e2world_{enc}"


def pad_dir(enc, d):
    return OUT / "pad" / enc / d


def code_shas():
    return {"code_sha256": {Path(f).name: Z2.sha_norm(f) for f in (__file__, HOOK / "sitecustomize.py", S.__file__)}}


# ── pad: the arm's vectors in the package's layout, 1536 wide ────────────────

def pad_shard(a, dim):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != dim or a.dtype != np.float16:
        stop(f"a shard {a.shape} {a.dtype}, not (rows, {dim}) float16")
    if not np.isfinite(a).all():
        stop("a shard holds a non-finite value")
    nrm = np.linalg.norm(a.astype(np.float32), axis=1)
    if a.shape[0] and float(np.abs(nrm - 1.0).max()) > 0.01:
        stop(f"a shard's rows are not unit length (max |norm-1| {float(np.abs(nrm - 1.0).max()):.4f})")
    out = np.zeros((a.shape[0], DIM), np.float16)
    out[:, :dim] = a
    return out


def save_npy(p, a):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, a)
    os.replace(tmp, p)
    return E.sha256(p)


def cmd_pad(a):
    enc, d = a.enc, a.ds
    if enc not in ARMS[1:] or d not in SMALL:
        stop(f"pad runs on {ARMS[1:]} and {SMALL}")
    root = E.served_root()
    dim = int(E.ENCODERS[enc]["dim"])
    out = pad_dir(enc, d)
    rec = {"stage": "ENC E2: pad", "declared_in": DECLARED, "dataset": d, "enc": enc, "dim_real": dim, "dim": DIM,
           "kinds": {}}
    for kind in ("docs", "queries"):
        man = E.manifest(root, d, kind)
        src = E.vec_shards(root, enc, d, kind)
        od = out / "embeddings" / "dense" / kind
        shards = []
        for s, p in enumerate(src):
            x = pad_shard(np.load(p, mmap_mode="r"), dim)
            want = man["shards"][s]["rows"]
            if x.shape[0] != want:
                stop(f"{enc} {d} {kind} shard {s}: {x.shape[0]} rows, the package's has {want}")
            fn = man["format"]["file"] % s
            sha = save_npy(od / fn, x)
            shards.append({"shard": s, "rows": int(x.shape[0]), "file": fn, "bytes": (od / fn).stat().st_size,
                           "sha256": sha, "source": Z2.rel(p), "source_sha256": E.sha256(p)})
            del x
        mf = dict(man, shards=shards)
        mf["encoder"] = {"encoder": E.ENCODERS[enc]["repo"], "revision": E.pinned(enc), "dim": DIM, "dim_real": dim,
                         "padding": f"zeros in columns {dim}..{DIM - 1}", "dtype": "float16"}
        mf["e2"] = {"declared_in": DECLARED, "source_manifest_sha256": E.sha256(E.vec_dir(root, enc, d, kind) / "manifest.json")}
        E.wjson(od / "manifest.json", mf)
        rec["kinds"][kind] = {"rows": int(man["n_rows"]), "manifest_sha256": E.sha256(od / "manifest.json")}
        log(f"{enc} {d} {kind}: {man['n_rows']:,} rows padded to {DIM}")
    fp = Z2.first_path(enc, d)
    fr = json.loads(fp.with_suffix(".json").read_text(encoding="utf-8"))
    if E.sha256(fp) != fr["npz_sha256"] or fr["ge0"]["verdict"] != "PASS":
        stop(f"{Z2.rel(fp)} is not its record's file, or its GE0 is not PASS")
    with np.load(fp) as z:
        ids, sc = np.asarray(z["ids"], np.int32), np.asarray(z["scores"], np.float16)
    nq = int(E.manifest(root, d, "queries")["n_rows"])
    if ids.shape != (nq, E.K) or sc.shape != ids.shape:
        stop(f"{Z2.rel(fp)}: lists {ids.shape}, the package has {nq} queries")
    cp = out / "retrieval_cache" / "dense_top1000.npz"
    rec["cache"] = {"file": Z2.rel(cp), "sha256": Z2.save_npz(cp, ids=ids, scores=sc), "from": Z2.rel(fp),
                    "from_sha256": fr["npz_sha256"], "ge0": fr["ge0"]["verdict"]}
    kp = Z2.knn_path(enc, d)
    kr = json.loads(kp.with_suffix(".json").read_text(encoding="utf-8"))
    if E.sha256(kp) != kr["npz_sha256"]:
        stop(f"{Z2.rel(kp)} is not its record's file")
    rec["knn"] = {"file": Z2.rel(kp), "sha256": kr["npz_sha256"]}
    rec.update(code_shas(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    E.wjson(out / "pad.json", rec)
    log(f"{enc} {d}: pad.json written")
    return 0


# ── the world ─────────────────────────────────────────────────────────────────

def configure():
    """sworld's make with E2's own folders: no PKO target (the world's pko is its own) and no link to the record's pko."""
    S.OWN = OWN
    S.PKO_TARGET = "outputs/enc/e2/__no_pko_target__"
    S.EXTRA_LINKS = {}


def deep_own(world, root):
    made = []
    for parent, own in DEEP.items():
        src, dst = Path(root) / "outputs" / parent, Path(world) / "outputs" / parent
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.iterdir()) if src.exists() else []:
            t = dst / p.name
            if p.name in own:
                if S.is_link(t):
                    stop(f"{t} is a junction; an own folder must be the world's")
                t.mkdir(exist_ok=True)
            elif p.is_dir():
                if S.junction(t, p):
                    made.append(f"outputs/{parent}/{p.name}")
            elif p.is_file() and not t.exists():
                if p.stat().st_size > COPY_CAP:
                    stop(f"{p} is over {COPY_CAP} bytes: not copied into a world")
                shutil.copy2(p, t)
        for name in own:
            (dst / name).mkdir(exist_ok=True)
    return made


def make_world(enc, root=ROOT, world=None):
    if enc not in ARMS:
        stop(f"the arm is one of {ARMS}")
    world = world_dir(enc, root) if world is None else Path(world)
    configure()
    S.make(world, root)
    made = deep_own(world, root)
    log(f"world {world.name}: made; {len(made)} deep junctions")
    return world


def served_str():
    return str(E.served_root())


def spec_of(enc, world, audit, root=ROOT, served=None):
    """The hook's spec: what the arm's world swaps, refuses and logs."""
    served = Path(served_str() if served is None else served)
    spec = {"declared_in": DECLARED, "enc": enc, "world": str(world), "root": str(root), "served": str(served), "audit": str(audit),
            "write_ok": [str(world), str(audit)], "swap": {}, "forbid": []}
    if enc == "frozen":
        return spec
    for d in SMALL:
        pd = pad_dir(enc, d)
        if not (pd / "pad.json").is_file():
            stop(f"{enc} {d}: no pad.json (python outputs/mp_unified/zenc2w.py pad --enc {enc} --ds {d})")
        spec["swap"][d] = {"docs": str(pd / "embeddings" / "dense" / "docs"),
                           "queries": str(pd / "embeddings" / "dense" / "queries"),
                           "cache": str(pd / "retrieval_cache" / "dense_top1000.npz"),
                           "knn": str(Z2.knn_path(enc, d))}
        spec["forbid"] += [str(served / d / "embeddings" / "dense"), str(served / d / "retrieval_cache" / "dense_top1000.npz"),
                           str(served / d / "graph" / "knn.npz"),
                           str(Path(root) / "outputs" / "m3b" / "csr" / f"{d}_knn.npz")]
    return spec


def cmd_world(a):
    make_world(a.enc)
    return 0


def run_in(enc, tag, cmd, root=ROOT, world=None, served=None, audit=None):
    world = world_dir(enc, root) if world is None else Path(world)
    if not (world / S.MANIFEST).is_file():
        stop(f"no world {world} (python outputs/mp_unified/zenc2w.py world --enc {enc})")
    S.verify(world, root, quiet=True)
    audit = OUT / "audit" / enc / tag if audit is None else Path(audit)
    if audit.exists() and any(audit.iterdir()):
        stop(f"{audit} holds a run already: pick another tag")
    audit.mkdir(parents=True, exist_ok=True)
    spec = spec_of(enc, world, audit, root, served)
    sp = audit / "spec.json"
    sp.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    env = dict(os.environ, E2HOOK=str(sp), E2WORLD=enc,
               PYTHONPATH=os.pathsep.join([str(world / "outputs" / "mp_unified" / "e2hook")]
                                          + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    log(f"{enc}/{tag}: {' '.join(cmd)}")
    t = time.time()
    rc = subprocess.run(cmd, cwd=str(world), env=env, check=False).returncode
    (audit / "run.json").write_text(json.dumps({"cmd": cmd, "rc": rc, "seconds": round(time.time() - t, 1),
                                                **code_shas()}, indent=1), encoding="utf-8")
    log(f"{enc}/{tag}: rc {rc} in {time.time() - t:.0f} s")
    rep = audit_report(audit)                      # written also when CMD failed: the audit is what a first run is for
    log(f"{enc}/{tag}: {rep['processes']} processes; blocked reads {len(rep['blocked_read'])}, blocked writes "
        f"{len(rep['blocked_write'])}, swaps {len(rep['swapped'])}; record reads {sum(rep['reads'].get('record', {}).values())}")
    return rc


def cmd_run(a, rest):
    if not rest:
        stop("run --enc E --tag T -- CMD")
    return run_in(a.enc, a.tag, rest)


# ── audit: what a run read ───────────────────────────────────────────────────

def norm(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def classify(p, world, served, root):
    w, s, r, o = norm(os.path.realpath(world)), norm(served), norm(root), norm(OUT)
    if p.startswith(w + "/"):
        return "world"
    if p.startswith(o + "/"):
        return "arm"
    if p.startswith(s + "/"):
        return "package"
    if p.endswith((".py", ".pyd", ".dll", ".zip", ".so", ".pth")) or "/pylib/" in p:
        return "code"
    if p.startswith(r + "/outputs/"):
        return "record"
    if p.startswith(r + "/"):
        return "workspace"
    return "other"


def summarize(audit):
    audit = Path(audit)
    spec = json.loads((audit / "spec.json").read_text(encoding="utf-8"))
    out = {"processes": 0, "argv": [], "blocked_read": [], "blocked_write": [], "swapped": [], "reads": {}, "writes": {}}
    for f in sorted(audit.glob("*.log")):
        out["processes"] += 1
        for ln in f.read_text(encoding="utf-8").splitlines():
            tag, _, p = ln.partition("\t")
            if tag == "A":
                out["argv"].append(json.loads(p)[:6])
            elif tag == "X":
                out["blocked_read"].append(p)
            elif tag == "Y":
                out["blocked_write"].append(p)
            elif tag == "S":
                out["swapped"].append(p)
            elif tag in ("R", "W"):
                k = classify(p, spec["world"], spec["served"], spec.get("root", str(ROOT)))
                out["reads" if tag == "R" else "writes"].setdefault(k, set()).add(p)
    for key in ("reads", "writes"):
        out[key] = {k: sorted(v) for k, v in sorted(out[key].items())}
    for key in ("blocked_read", "blocked_write", "swapped"):
        out[key] = sorted(set(out[key]))
    out["root"] = spec.get("root", str(ROOT))
    return out


def group(paths, root, depth=4):
    r = norm(root)
    g = {}
    for p in paths:
        q = p[len(r) + 1:] if p.startswith(r + "/") else p
        k = "/".join(q.split("/")[:depth])
        g[k] = g.get(k, 0) + 1
    return dict(sorted(g.items()))


def audit_report(audit):
    s = summarize(audit)
    rep = {"processes": s["processes"], "blocked_read": s["blocked_read"], "blocked_write": s["blocked_write"],
           "swapped": s["swapped"], "reads": {k: group(v, s["root"]) for k, v in s["reads"].items() if k != "code"},
           "record_reads": s["reads"].get("record", []), "package_reads": s["reads"].get("package", []),
           "writes": {k: group(v, s["root"]) for k, v in s["writes"].items()}, "argv": s["argv"][:20]}
    E.wjson(Path(audit) / "summary.json", rep)
    return rep


def cmd_audit(a):
    rep = audit_report(OUT / "audit" / a.enc / a.tag)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("record_reads", "package_reads")}, indent=1))
    return 0


# ── self-test ─────────────────────────────────────────────────────────────────

CANON_STUB = '''import io, json, os
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
def _rj(p):
    return json.load(io.open(p, encoding="utf-8"))
class Embeddings(object):
    def __init__(self, d, model):
        self.dir, self.model = d, model
        self.manifest = _rj(os.path.join(d, "manifest.json"))
    def read(self, rows):
        return np.load(os.path.join(self.dir, "shard_00000.npy"), mmap_mode="r")[np.asarray(rows)]
class Dataset(object):
    def __init__(self, name, root=None):
        self.name, self.root = name, root or _HERE
        self.dir = os.path.join(self.root, name)
    def embeddings(self, model, kind="docs"):
        return Embeddings(os.path.join(self.dir, "embeddings", model, kind), model)
    def cache(self, model):
        z = np.load(os.path.join(self.dir, "retrieval_cache", "%s_top1000.npz" % model))
        return z["ids"], z["scores"]
    @property
    def graph_manifest(self):
        return _rj(os.path.join(self.dir, "graph", "GRAPH_MANIFEST.json"))
class Graph(object):
    def __init__(self, ds, family):
        self.manifest = ds.graph_manifest
        z = np.load(os.path.join(ds.dir, "graph", family + ".npz"))
        self.src, self.dst = z["src"], z["dst"]
        self.provenance = self.manifest["families"][family]
'''

CHILD = '''import os, sys, json
import numpy as np
served, world, outside = sys.argv[1:4]
sys.path.insert(0, served)
import canonical as C
ds = C.Dataset("squad")
e = ds.embeddings("dense", "docs")
q = ds.embeddings("dense", "queries")
ids, sc = ds.cache("dense")
r = {"docs": e.dir, "queries": q.dir, "row": e.read([1])[0, :4].tolist(), "ids0": int(ids[0, 0]),
     "other_ds": C.Dataset("hotpotqa").embeddings("dense", "docs").dir}
g = C.Graph(ds, "knn")
r["knn"] = [g.src.tolist(), g.dst.tolist(), g.provenance["n_edges"], str(g.src.dtype)]
r["knn_other"] = C.Graph(C.Dataset("hotpotqa"), "knn").src.tolist()
try:
    np.load(os.path.join(served, "squad", "embeddings", "dense", "docs", "shard_00000.npy"))
    r["direct"] = "read"
except PermissionError:
    r["direct"] = "refused"
try:
    open(os.path.join(outside, "w.txt"), "w").write("x")
    r["write_out"] = "written"
except PermissionError:
    r["write_out"] = "refused"
open(os.path.join(world, "ok.txt"), "w").write("x")
r["write_in"] = "written"
print("RESULT " + json.dumps(r))
'''


def selftest():
    tmp = Path(tempfile.mkdtemp(prefix="zenc2w_t_"))
    try:
        # the hook, in a child process, on a stub loader
        served, world, outside, audit = tmp / "served", tmp / "world", tmp / "outside", tmp / "audit"
        for p in (world, outside):
            p.mkdir(parents=True)
        (served / "canonical.py").parent.mkdir(parents=True)
        (served / "canonical.py").write_text(CANON_STUB, encoding="utf-8")
        rec_rows = np.eye(4, DIM, dtype=np.float16)
        arm_rows = pad_shard((np.eye(4, 768, k=1) + 0).astype(np.float16), 768)
        for ds in ("squad", "hotpotqa"):
            for kind in ("docs", "queries"):
                d = served / ds / "embeddings" / "dense" / kind
                d.mkdir(parents=True)
                np.save(d / "shard_00000.npy", rec_rows)
                (d / "manifest.json").write_text(json.dumps({"n_rows": 4}), encoding="utf-8")
            (served / ds / "graph").mkdir(parents=True)
            np.savez(served / ds / "graph" / "knn.npz", src=np.array([0], np.int32), dst=np.array([1], np.int32))
            (served / ds / "graph" / "GRAPH_MANIFEST.json").write_text(json.dumps(
                {"n_nodes": 4, "families": {"knn": {"present": True, "directed": False, "n_edges": 1}}}), encoding="utf-8")
            (served / ds / "retrieval_cache").mkdir(parents=True)
            np.savez(served / ds / "retrieval_cache" / "dense_top1000.npz", ids=np.zeros((2, 1000), np.int32),
                     scores=np.zeros((2, 1000), np.float16))
        arm = tmp / "arm" / "squad"
        for kind in ("docs", "queries"):
            d = arm / "embeddings" / "dense" / kind
            d.mkdir(parents=True)
            np.save(d / "shard_00000.npy", arm_rows)
            (d / "manifest.json").write_text(json.dumps({"n_rows": 4}), encoding="utf-8")
        (arm / "retrieval_cache").mkdir(parents=True)
        np.savez(arm / "retrieval_cache" / "dense_top1000.npz", ids=np.full((2, 1000), 3, np.int32),
                 scores=np.ones((2, 1000), np.float16))
        spec = {"enc": "mpnet", "world": str(world), "served": str(served), "audit": str(audit),
                "write_ok": [str(world), str(audit)], "temp_ok": False,
                "swap": {"squad": {"docs": str(arm / "embeddings/dense/docs"), "queries": str(arm / "embeddings/dense/queries"),
                                   "cache": str(arm / "retrieval_cache/dense_top1000.npz"), "knn": str(tmp / "arm_knn.npz")}},
                "forbid": [str(served / "squad" / "embeddings" / "dense"), str(served / "squad" / "graph" / "knn.npz"),
                           str(served / "squad" / "retrieval_cache" / "dense_top1000.npz")]}
        np.savez(tmp / "arm_knn.npz", src=np.array([2, 0], np.int64), dst=np.array([3, 3], np.int64),
                 weight=np.array([0.5, 0.25], np.float64))
        sp = tmp / "spec.json"
        sp.write_text(json.dumps(spec), encoding="utf-8")
        (tmp / "child.py").write_text(CHILD, encoding="utf-8")
        env = dict(os.environ, E2HOOK=str(sp), PYTHONPATH=str(HOOK))
        r = subprocess.run([sys.executable, str(tmp / "child.py"), str(served), str(world), str(outside)], env=env,
                           capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
        assert r.returncode == 0 and line, (r.returncode, r.stdout[-800:], r.stderr[-1500:])
        got = json.loads(line[0][7:])
        assert norm(got["docs"]) == norm(arm / "embeddings/dense/docs"), got
        assert norm(got["queries"]) == norm(arm / "embeddings/dense/queries"), got
        assert got["row"] == [0.0, 0.0, 1.0, 0.0] and got["ids0"] == 3, got
        assert norm(got["other_ds"]) == norm(served / "hotpotqa/embeddings/dense/docs"), got
        assert got["knn"] == [[2, 0], [3, 3], 2, "int32"] and got["knn_other"] == [0], got
        assert got["direct"] == "refused" and got["write_out"] == "refused" and got["write_in"] == "written", got
        logs = list(audit.glob("*.log"))
        assert len(logs) == 1
        tags = {ln.split("\t")[0] for ln in logs[0].read_text(encoding="utf-8").splitlines()}
        assert {"A", "R", "W", "X", "Y", "S"} <= tags, tags
        # without E2HOOK the hook does nothing: the record's rows are read
        env0 = dict(os.environ, PYTHONPATH=str(HOOK))
        env0.pop("E2HOOK", None)
        r0 = subprocess.run([sys.executable, str(tmp / "child.py"), str(served), str(world), str(outside)], env=env0,
                            capture_output=True, text=True)
        got0 = json.loads([x for x in r0.stdout.splitlines() if x.startswith("RESULT ")][0][7:])
        assert got0["direct"] == "read" and got0["write_out"] == "written" and got0["ids0"] == 0, got0
        # pad keeps every inner product
        x = np.random.default_rng(0).standard_normal((50, 768)).astype(np.float32)
        x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
        px = pad_shard(x, 768)
        assert px.shape == (50, DIM) and not px[:, 768:].any()
        g_pad = px.astype(np.float32) @ px.astype(np.float32).T      # the same sums up to float rounding (BLAS order)
        assert float(np.abs(g_pad - x.astype(np.float32) @ x.astype(np.float32).T).max()) < 1e-5
        try:
            pad_shard((x * 2).astype(np.float16), 768)
            raise AssertionError("pad must refuse rows that are not unit length")
        except SystemExit:
            pass
        # the world: sworld's make with E2's own folders and deep_own
        root, w2 = tmp / "ws", tmp / "e2world_mpnet"
        for d, f in (("src", "a.py"), ("scripts", "b.py"), ("docs", "c.md"), ("configs", "d.json"), ("outputs/mp_unified", "m.py"),
                     ("outputs/host_ops", "h.py"), ("outputs/abl/D83F", "x.txt"), ("outputs/abl/NKF", "y.txt"),
                     ("outputs/m3b/csr", "squad_knn.npz"), ("outputs/pko/pools", "p.txt"), ("data", "z.txt")):
            (root / d).mkdir(parents=True, exist_ok=True)
            (root / d / f).write_text("x", encoding="utf-8")
        (root / "outputs/abl/gate.json").write_text("{}", encoding="utf-8")
        (root / "outputs/mp_unified/e2hook").mkdir()
        shutil.copy2(HOOK / "sitecustomize.py", root / "outputs/mp_unified/e2hook/sitecustomize.py")
        make_world("mpnet", root, w2)
        real = os.path.realpath
        assert (w2 / "outputs/pko").is_dir() and not S.is_link(w2 / "outputs/pko") and not any((w2 / "outputs/pko").iterdir())
        assert (w2 / "outputs/abl/D83F").is_dir() and not S.is_link(w2 / "outputs/abl/D83F")
        assert not any((w2 / "outputs/abl/D83F").iterdir())
        assert S.is_link(w2 / "outputs/abl/NKF") and real(w2 / "outputs/abl/NKF") == real(root / "outputs/abl/NKF")
        assert (w2 / "outputs/abl/gate.json").is_file() and not S.is_link(w2 / "outputs/abl")
        assert S.is_link(w2 / "outputs/m3b") and S.is_link(w2 / "data")
        S.verify(w2, root, quiet=True)
        make_world("mpnet", root, w2)                       # a refresh keeps the own folders
        sp2 = spec_of("frozen", w2, tmp / "a2", root, served)
        assert sp2["swap"] == {} and sp2["forbid"] == [] and str(w2) in sp2["write_ok"]
        # run: verify, the hook from the world's e2hook, the audit summary also on a failed command
        code = "import os;open('made.txt','w').write('1');open(os.path.join('data','z.txt')).read();raise SystemExit(5)"
        rc = run_in("frozen", "t1", [sys.executable, "-c", code], root, w2, served, tmp / "a3")
        rep = json.loads((tmp / "a3" / "summary.json").read_text(encoding="utf-8"))
        assert rc == 5 and rep["processes"] == 1 and rep["blocked_read"] == [] and rep["blocked_write"] == [], rep
        assert (w2 / "made.txt").is_file() and rep["reads"].get("workspace") == {"data/z.txt": 1}, rep
        assert list(rep["writes"]) == ["world"], rep
        try:
            run_in("frozen", "t1", [sys.executable, "-c", "pass"], root, w2, served, tmp / "a3")
            raise AssertionError("run must refuse a tag that holds a run")
        except SystemExit:
            pass
    finally:
        for p in sorted(tmp.rglob("*"), key=lambda q: -len(q.parts)):
            if S.is_link(p):
                os.rmdir(p) if os.name == "nt" else p.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
    print("zenc2w self-test: OK")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
        argv = argv[:argv.index("--")]
    ap = argparse.ArgumentParser(prog="zenc2w.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pad")
    p.add_argument("--enc", required=True)
    p.add_argument("--ds", required=True)
    p.add_argument("--threads", type=int, default=0)
    p = sub.add_parser("world")
    p.add_argument("--enc", required=True)
    p = sub.add_parser("run")
    p.add_argument("--enc", required=True)
    p.add_argument("--tag", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--enc", required=True)
    p.add_argument("--tag", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "pad":
        return cmd_pad(a)
    if a.cmd == "world":
        return cmd_world(a)
    if a.cmd == "run":
        return cmd_run(a, rest)
    if a.cmd == "audit":
        return cmd_audit(a)
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
