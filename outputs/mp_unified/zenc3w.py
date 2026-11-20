"""ENC E3W (docs/ENC_ENCODER_COMPARISON.md section 19): E2's modules (sections 12.9 to 12.11), unchanged, run on the
fused arm e3v (zenc3v.py).

    python outputs/mp_unified/zenc3w.py MODULE ARGS...    MODULE: zenc2, zenc1b2, zenc2w, zenc2p, zenc2k or zenc2t
    python outputs/mp_unified/zenc3w.py mark              the e3v world's own D83 folders get NTFS compression (19.7)
    python outputs/mp_unified/zenc3w.py selftest [--real]

What it changes, in this process only (E2's files are not edited, so their records keep their code shas):
- e3v joins each module's arm list: zenc2.CARRY, zenc1b2.ARMS, zenc2w.ARMS, zenc2k.ARMS, zenc.NEW (zenc2p's bases),
  and zenc2t's --enc choices.
- zenc2p's bases (19.3, GB): the drawn rows of a graph e3v has no vectors of are encoded by gtemodern and granitesm as
  E0N encoded (zenc's load_model and encode_texts: float16 on the GPU, each encoder's own token budget) and fused as e3v
  is (zenc3v.fuse).
The worlds, the hook, the pools, the chains, the fits and the reads are E2's code as it ran for mpnet and qwen3e06. The
in-world rows call E2's modules directly (zenc2w run and its hook take the arm from the spec, not from a list).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODS = ("zenc2", "zenc1b2", "zenc2w", "zenc2p", "zenc2k", "zenc2t")
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
CMD = sys.argv[1]
if CMD in MODS:
    sys.argv = [str(HERE / f"{CMD}.py")] + sys.argv[2:]   # zenc reads --threads from here at import, before numpy loads
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zenc as E  # noqa: E402
import zenc3v as V  # noqa: E402  (registers e3v)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

COMPRESSED = 0x800                                    # FILE_ATTRIBUTE_COMPRESSED
REPARSE = 0x400                                       # FILE_ATTRIBUTE_REPARSE_POINT
BATCH = 24


def log(msg):
    print(f"[zenc3w {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc3w: {msg}")


# ── the patches ──────────────────────────────────────────────────────────────

_add_argument = argparse._ActionsContainer.add_argument


def _add_argument_e3v(self, *args, **kw):
    """An --enc whose choices are E2's arms takes e3v too (zenc2t's)."""
    ch = kw.get("choices")
    if ch is not None and not isinstance(ch, (str, bytes)) and "qwen3e06" in tuple(ch) and V.ENC not in tuple(ch):
        kw["choices"] = tuple(ch) + (V.ENC,)
    return _add_argument(self, *args, **kw)


def widen(mod, name):
    v = tuple(getattr(mod, name))
    if V.ENC not in v:
        setattr(mod, name, v + (V.ENC,))


def encode_part(e, texts, dev):
    """One part's float16 unit vectors of texts, as E0N encoded (zenc3.cmd_encode -> zenc.cmd_encode's encode_texts)."""
    import torch
    t = time.time()
    rev = E.pinned(e)
    m = E.load_model(e, dev, rev)
    if m.default_prompt_name:
        stop(f"{e}: a default prompt would prefix the documents")
    v, st = E.encode_texts(m, e, texts, "docs")
    bad = E.damaged(v)
    if bad:
        stop(f"{e}: {bad} damaged rows")
    info = {"revision": rev, "device": dev, "dtype": "float16" if dev == "cuda" else "float32 (stored float16)",
            "max_seq_length": int(m.max_seq_length), "token_budget": int(E.ENCODERS[e]["budget"]),
            "params": int(sum(p.numel() for p in m.parameters())), "encode_seconds": round(time.time() - t, 1), **st}
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return v, info


def fused_rows(g, rows, torch_threads, budget=None):
    """zenc2p.encoded_rows for e3v: each part encodes the drawn rows on the GPU, then fuse (19.3, GB)."""
    import torch
    torch.set_num_threads(int(torch_threads))
    if not torch.cuda.is_available():
        stop("no GPU: E0N encoded on the GPU, so e3v's basis rows are encoded there too (19.3)")
    root = E.served_root()
    t = time.time()
    texts = E.doc_texts(root, g, rows)
    if len(texts) != rows.size:
        stop(f"{g}: {len(texts)} texts for {rows.size} rows")
    seq = [texts[int(r)] for r in rows]
    vs, info = [], {}
    for e in V.PARTS:
        v, info[e] = encode_part(e, seq, "cuda")
        vs.append(v)
        log(f"bases {g}: {e} encoded {rows.size:,} rows in {info[e]['encode_seconds']} s")
    X = V.fuse(vs)
    bad = E.damaged(X)
    if bad:
        stop(f"{g}: {bad} damaged fused rows")
    return X, {"source": "the drawn rows' texts, encoded by each part as E0N encoded and fused as e3v (section 19.3)",
               "device": "cuda", "torch_threads": int(torch_threads), "revision": E.pinned(V.ENC), "fusion": V.FUSION,
               "cli_budget_unused": budget, "parts": info, "encode_seconds": round(time.time() - t, 1)}


def patch(name, mod):
    if name == "zenc2":
        widen(mod, "CARRY")
    elif name in ("zenc1b2", "zenc2w", "zenc2k"):
        widen(mod, "ARMS")
    elif name == "zenc2p":
        widen(E, "NEW")
        orig = mod.encoded_rows

        def encoded_rows(enc, g, rows, torch_threads, budget):
            if enc != V.ENC:
                return orig(enc, g, rows, torch_threads, budget)
            return fused_rows(g, rows, torch_threads, budget)
        mod.encoded_rows = encoded_rows
    elif name == "zenc2t":
        import zenc2w
        widen(zenc2w, "ARMS")                          # zenc2t's z2w() returns this module (make_world checks ARMS)
    argparse._ActionsContainer.add_argument = _add_argument_e3v
    return mod


def run_module(name):
    mod = patch(name, importlib.import_module(name))
    rc = mod.main()
    return 0 if rc is None else rc


# ── mark: the world's own D83 folders compressed before the chains write ─────

def is_reparse(p):
    try:
        return os.path.islink(p) or bool(getattr(os.lstat(p), "st_file_attributes", 0) & REPARSE)
    except OSError:
        return True


def own_dirs(top):
    dirs = [Path(top)]
    for dp, dn, _fn in os.walk(top):
        dn[:] = [d for d in dn if not is_reparse(os.path.join(dp, d))]
        dirs += [Path(dp) / d for d in dn]
    return dirs


def cmd_mark():
    import zenc2w as Z2W
    world = Z2W.world_dir(V.ENC)
    tops = [world / "outputs" / "abl" / a for a in Z2W.DEEP["abl"]]
    rec = {"stage": V.STAGE + ": mark", "declared_in": V.DECLARED + ".7", "world": str(world), "folders": [],
           "utc": V.utc()}
    for t in tops:
        if not t.is_dir() or is_reparse(t):
            stop(f"{t} is not the world's own folder (make the world first)")
        dirs = own_dirs(t)
        files = sum(len(fn) for _dp, _dn, fn in os.walk(t))
        for i in range(0, len(dirs), BATCH):
            r = subprocess.run(["compact", "/c", "/q", "/i"] + [str(d) for d in dirs[i:i + BATCH]],
                               capture_output=True, text=True)
            if r.returncode != 0:
                stop(f"compact returned {r.returncode} on {t}: {r.stdout[-300:]} {r.stderr[-300:]}")
        miss = [str(d) for d in dirs if not (getattr(os.stat(d), "st_file_attributes", 0) & COMPRESSED)]
        if miss:
            stop(f"{len(miss)} folders are not marked compressed ({miss[:3]})")
        rec["folders"].append({"folder": str(t), "dirs": len(dirs), "files_present": files})
        log(f"marked {t} ({len(dirs)} folders, {files} files present): new files there are compressed as written")
    V.OUT.mkdir(parents=True, exist_ok=True)
    E.wjson(V.OUT / "mark.json", rec)
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def selftest(real):
    V.selftest_toy()
    argparse._ActionsContainer.add_argument = _add_argument_e3v
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", choices=("frozen", "mpnet", "qwen3e06"))
    ap.add_argument("--where", choices=("train", "e2"))
    assert ap.parse_args(["--enc", V.ENC]).enc == V.ENC
    try:
        ap.parse_args(["--where", V.ENC])
        raise AssertionError("an unrelated choice took e3v")
    except SystemExit:
        pass
    for name, attr in (("zenc2", "CARRY"), ("zenc1b2", "ARMS"), ("zenc2w", "ARMS"), ("zenc2k", "ARMS")):
        mod = patch(name, importlib.import_module(name))
        assert V.ENC in getattr(mod, attr), (name, attr)
        n = len(getattr(mod, attr))
        patch(name, mod)
        assert len(getattr(mod, attr)) == n, f"{name}.{attr} widened twice"
    P = patch("zenc2p", importlib.import_module("zenc2p"))
    assert V.ENC in E.NEW and P.encoded_rows.__name__ == "encoded_rows" and P.encoded_rows is not fused_rows
    import zenc2w
    assert zenc2w.world_dir(V.ENC).name == "e2world_e3v" and zenc2w.pad_dir(V.ENC, "squad").parts[-2:] == ("e3v", "squad")
    with __import__("tempfile").TemporaryDirectory() as td:
        d = Path(td) / "a" / "b"
        d.mkdir(parents=True)
        assert [p.name for p in own_dirs(Path(td) / "a")] == ["a", "b"]
    print("zenc3w toy self-test: OK")
    if not real:
        return 0
    V.selftest_real()
    # GB on the CPU: the encode path gives E0N's stored vectors back (cosine >= 0.99, zenc3's GP floor) on 16 SQuAD docs
    import torch
    torch.set_num_threads(4)
    root = E.served_root()
    rows = np.arange(16, dtype=np.int64) * 997
    texts = E.doc_texts(root, "squad", rows)
    params = {}
    for e in V.PARTS:
        v, info = encode_part(e, [texts[int(r)] for r in rows], "cpu")
        stored = np.concatenate([np.load(p, mmap_mode="r") for p in E.vec_shards(root, e, "squad", "docs")])[rows]
        cos = np.sum(v.astype(np.float32) * stored.astype(np.float32), axis=1)
        log(f"real GB {e}: lowest cosine to E0N's stored vector {float(cos.min()):.5f}; {info['params']:,} parameters")
        assert float(cos.min()) >= 0.99, (e, float(cos.min()))
        params[e] = info["params"]
    E.wjson(V.OUT / "params.json", {"stage": V.STAGE, "declared_in": V.DECLARED, "params": params,
                                    "total": int(sum(params.values())), "utc": V.utc()})
    print("zenc3w real self-test: OK")
    return 0


def main():
    if CMD in MODS:
        return run_module(CMD)
    if CMD == "mark":
        return cmd_mark()
    if CMD == "selftest":
        return selftest("--real" in sys.argv[2:])
    stop(f"unknown command {CMD!r}: one of {MODS}, mark or selftest")


if __name__ == "__main__":
    raise SystemExit(main())
